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


class IdentityOneBitSTE(nn.Module):
    """
    Starts EXACTLY at the baseline: one_bit_quantization(base_embeddings).
    Achieved by identity projection + L2 normalize + STE sign.
    """
    def __init__(self, in_dim: int, out_dim: int, use_ste: bool = True, temperature: float = 1.0):
        super().__init__()
        # Identity-capable projection (in_dim -> out_dim)
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        self._init_identity(in_dim, out_dim)
        self.use_ste = use_ste
        self.temperature = temperature

    def _init_identity(self, in_dim: int, out_dim: int) -> None:
        with torch.no_grad():
            # Tiny noise everywhere to encourage gradient flow on non-identity entries
            noise_std = 1e-3
            nn.init.normal_(self.proj.weight, mean=0.0, std=noise_std)
            nn.init.zeros_(self.proj.bias)
            # Place an exact identity block of size k=min(in_dim, out_dim) on top of the noise
            k = min(in_dim, out_dim)
            idx = torch.arange(k)
            self.proj.weight[idx, idx] = 1.0  # identity on diagonals; off-diagonals keep tiny noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x+ self.proj(x)  # identity at init
        z = torch.nn.functional.normalize(z, p=2, dim=1)  # baseline preprocess
        if self.use_ste:
            s = torch.sign(z)
            q = z + (s - z).detach()
        else:
            q = torch.tanh(z / self.temperature)
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
embed_dim = model.get_sentence_embedding_dimension()
config = {
    "base_model": "msmarco-bert-base-dot-v5",
    "in_dim": embed_dim,
    "out_dim": embed_dim,          # set > in_dim to expand; identity block fills min(in,out)
    "use_ste": False,  # Use tanh instead of STE
    "proj_temperature": 1.0,
    "loss_temperature": 0.05,
    "batch_size": 4096,
    "learning_rate": 5e-4,
    "num_epochs": 5,
    "optimizer": "Adam",
    "loss": "InfoNCE",
}

# Initialize wandb
wandb.init(
    project="quantized-mlp-training",
    config=config,
    name="identity-onebit-tanh"
)

quantizer = IdentityOneBitSTE(
    in_dim=config["in_dim"],
    out_dim=config["out_dim"],
    use_ste=config["use_ste"],
    temperature=config["proj_temperature"],
).to(device)
optimizer = torch.optim.Adam(quantizer.parameters(), lr=config["learning_rate"])

wandb.config.update({
    "quantizer_params": sum(p.numel() for p in quantizer.parameters()),
    "quantizer_trainable": sum(p.numel() for p in quantizer.parameters() if p.requires_grad),
})


def infoNCE_loss(similarity_matrix: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    batch_size = similarity_matrix.size(0)
    logits = similarity_matrix / temperature
    labels = torch.arange(batch_size, device=logits.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss
eval_dataset = create_eval_dataset(max_queries=None, use_scoreddocs=False)
eval_loader = DataLoader(eval_dataset, batch_size=16384, shuffle=False, collate_fn=in_batch_negatives_collate_fn)
print(f"number of batches in eval: {len(eval_loader)}")

quantizer.train()
global_step = 0

with torch.no_grad():
    tmp_sentences = ["hello world", "this is a test", "another query"]
    tmp_q = model.encode(tmp_sentences, convert_to_tensor=True, device=device)
    base_bin = one_bit_quantization(tmp_q)
    learned = one_bit_quantization(quantizer(tmp_q))
    exact_match = torch.all((base_bin == learned))
    print(f"Sanity check - exact baseline match at init: {bool(exact_match)}")

results_quantized = {"correct": 0, "total": 0}
results_baseline = {"correct": 0, "total": 0}
quantizer.eval()
with torch.no_grad():
    for batch in eval_loader:
        queries = [example.texts[0] for example in batch]
        documents = batch[0].texts[1:]

        query_embeddings = model.encode(queries, convert_to_tensor=True, device=device)
        document_embeddings = model.encode(documents, convert_to_tensor=True, device=device)
        baseline_q = one_bit_quantization(query_embeddings)
        baseline_d = one_bit_quantization(document_embeddings)
        sim_base = baseline_q @ baseline_d.T
        for i in range(len(batch)):
            best = torch.argmax(sim_base[i, :])
            if best == i:
                results_baseline["correct"] += 1
            results_baseline["total"] += 1

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
print(f"Epoch 0 eval — Baseline (direct 1-bit): {acc_base:.5f}")
print(f"Epoch 0 eval — Learned (identity+tanh 1-bit): {acc_learned:.5f}")
wandb.log({
    "eval/accuracy_baseline_onebit": acc_base,
    "eval/accuracy_learned_onebit": acc_learned,
    "eval/epoch": 0,
})
quantizer.train()

for epoch in range(config["num_epochs"]):
    epoch_losses = []

    for batch in train_loader:
        queries = [example.texts[0] for example in batch]
        documents = batch[0].texts[1:]

        with torch.no_grad():
            query_embeddings = model.encode(queries, convert_to_tensor=True, device=device)
            document_embeddings = model.encode(documents, convert_to_tensor=True, device=device)

        query_embeddings = query_embeddings.clone()
        document_embeddings = document_embeddings.clone()

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

            # Learned 1-bit: identity + tanh 1-bit
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
    print(f"Epoch {epoch+1} eval — Learned (identity+tanh 1-bit): {acc_learned:.5f}")
    wandb.log({
        "eval/accuracy_baseline_onebit": acc_base,
        "eval/accuracy_learned_onebit": acc_learned,
        "eval/epoch": epoch + 1,
    })
    quantizer.train()

print(f"Training completed. Final epoch loss: {avg_loss:.4f}")

torch.save(quantizer.state_dict(), "identity_onebit_tanh.pth")

wandb.finish()


