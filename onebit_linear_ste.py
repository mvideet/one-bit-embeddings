import torch
import torch.nn as nn
import os
os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER', None)
import wandb
from sentence_transformers import SentenceTransformer
from dataset import create_eval_dataset, in_batch_negatives_collate_fn, create_train_dataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')
model = model.to(device)
print(model)


class OneBitLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, use_ste: bool = True, temperature: float = 1.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=True)
        self.use_ste = use_ste
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        if self.use_ste:
            s = torch.sign(z)
            q = z + (s - z).detach()  # straight-through estimator
        else:
            q = torch.tanh(z / self.temperature)
        q = torch.nn.functional.normalize(q, p=2, dim=1)
        return q


def one_bit_quantization(embeddings: torch.Tensor) -> torch.Tensor:
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return torch.where(embeddings > 0, torch.ones_like(embeddings), -torch.ones_like(embeddings))


# Create dataset and dataloader for training
train_dataset = create_train_dataset(use_scoreddocs=False)
train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, collate_fn=in_batch_negatives_collate_fn)
print("loading data done")
print(f"number of batches in training: {len(train_loader)}")

# Hyperparameters
config = {
    "base_model": "msmarco-bert-base-dot-v5",
    "input_dim": model.get_sentence_embedding_dimension(),
    "output_dim": 1024,
    "use_ste": True,
    "proj_temperature": 1.0,
    "loss_temperature": 0.05,
    "batch_size": 4096,
    "learning_rate": 1e-3,
    "num_epochs": 5,
    "optimizer": "Adam",
    "loss": "InfoNCE",
}

# Initialize wandb
wandb.init(
    project="quantized-mlp-training",
    config=config,
    name=f"onebit-linear-ste-{config['output_dim']}"
)

quantizer = OneBitLinear(
    input_dim=config["input_dim"],
    output_dim=config["output_dim"],
    use_ste=config["use_ste"],
    temperature=config["proj_temperature"],
).to(device)
optimizer = torch.optim.Adam(quantizer.parameters(), lr=config["learning_rate"])

# Log model architecture
wandb.config.update({
    "quantizer_params": sum(p.numel() for p in quantizer.parameters()),
    "quantizer_trainable": sum(p.numel() for p in quantizer.parameters() if p.requires_grad),
})

# Print memory usage info
if torch.cuda.is_available():
    base_model_mb = sum(p.numel() * 4 for p in model.parameters()) / 1024**2
    q_mb = sum(p.numel() * 4 for p in quantizer.parameters()) / 1024**2
    print(f"\nGPU Memory Info:")
    print(f"  Base model: ~{base_model_mb:.1f} MB")
    print(f"  OneBitLinear: ~{q_mb:.1f} MB")


def infoNCE_loss(similarity_matrix: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    batch_size = similarity_matrix.size(0)
    logits = similarity_matrix / temperature
    labels = torch.arange(batch_size, device=logits.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


# Prepare full dev evaluation loader (run after every epoch)
eval_dataset = create_eval_dataset(max_queries=None, use_scoreddocs=False)
eval_loader = DataLoader(eval_dataset, batch_size=16384, shuffle=False, collate_fn=in_batch_negatives_collate_fn)
print(f"number of batches in eval: {len(eval_loader)}")


# Training loop
quantizer.train()
global_step = 0

for epoch in range(config["num_epochs"]):
    epoch_losses = []

    for batch in train_loader:
        queries = [example.texts[0] for example in batch]
        documents = batch[0].texts[1:]

        with torch.no_grad():
            query_embeddings = model.encode(queries, convert_to_tensor=True, device=device)
            document_embeddings = model.encode(documents, convert_to_tensor=True, device=device)

        # Train the linear quantizer
        query_vecs = quantizer(query_embeddings)
        doc_vecs = quantizer(document_embeddings)

        sim = query_vecs @ doc_vecs.T
        loss = infoNCE_loss(sim, temperature=config["loss_temperature"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        global_step += 1
        if global_step % 10 == 0:
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/epoch": epoch + 1,
                "train/global_step": global_step,
            })

    avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
    print(f"Epoch {epoch+1} loss: {avg_loss:.4f}")
    wandb.log({
        "train/epoch_loss": avg_loss,
        "train/epoch": epoch + 1,
    })

    # Per-epoch evaluation on full dev set
    results_quantized = {"correct": 0, "total": 0}
    results_baseline = {"correct": 0, "total": 0}
    quantizer.eval()
    with torch.no_grad():
        for batch in eval_loader:
            queries = [example.texts[0] for example in batch]
            documents = batch[0].texts[1:]

            query_embeddings = model.encode(queries, convert_to_tensor=True, device=device)
            document_embeddings = model.encode(documents, convert_to_tensor=True, device=device)

            # Baseline: direct 1-bit from base embeddings
            baseline_q = one_bit_quantization(query_embeddings)
            baseline_d = one_bit_quantization(document_embeddings)
            sim_base = baseline_q @ baseline_d.T
            for i in range(len(batch)):
                best = torch.argmax(sim_base[i, :])
                if best == i:
                    results_baseline["correct"] += 1
                results_baseline["total"] += 1

            # Learned 1-bit: linear + STE + 1-bit
            q_raw = quantizer(query_embeddings)
            d_raw = quantizer(document_embeddings)
            q_bin = one_bit_quantization(q_raw)
            d_bin = one_bit_quantization(d_raw)
            sim_learned = q_bin @ d_bin.T
            for i in range(len(batch)):
                best = torch.argmax(sim_learned[i, :])
                if best == i:
                    results_quantized["correct"] += 1
                results_quantized["total"] += 1

    acc_base = results_baseline['correct'] / max(1, results_baseline['total'])
    acc_learned = results_quantized['correct'] / max(1, results_quantized['total'])
    print(f"Epoch {epoch+1} eval — Baseline (direct 1-bit): {acc_base:.5f}")
    print(f"Epoch {epoch+1} eval — Learned (linear+STE 1-bit): {acc_learned:.5f}")
    wandb.log({
        "eval/accuracy_baseline_onebit": acc_base,
        "eval/accuracy_learned_onebit": acc_learned,
        "eval/epoch": epoch + 1,
    })
    quantizer.train()

print(f"Training completed. Final epoch loss: {avg_loss:.4f}")

torch.save(quantizer.state_dict(), "onebit_linear_ste.pth")

wandb.finish()


