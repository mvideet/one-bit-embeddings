import torch
import torch.nn as nn
import os
os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER', None)
import wandb
from sentence_transformers import SentenceTransformer
from dataset import create_eval_dataset, in_batch_negatives_collate_fn, create_train_dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')
model = model.to(device)
print(model)

class onebitMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, temperature=1.0, use_layernorm=True):
        super(onebitMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.ln = torch.nn.LayerNorm(output_dim) if use_layernorm else None
        self.temperature = temperature

    def forward(self, x):
        z = self.mlp(x)
        if self.ln is not None:
            z = self.ln(z)
        q = torch.tanh(z / self.temperature)
        q = torch.nn.functional.normalize(q, p=2, dim=1)
        return q


def one_bit_quantization(embeddings):
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    quantized = torch.where(embeddings > 0, torch.ones_like(embeddings), -torch.ones_like(embeddings))
    return quantized

# Create dataset and dataloader for training
train_dataset = create_train_dataset(use_scoreddocs=False)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=in_batch_negatives_collate_fn)
print("loading data done")
print(f"number of batches in training: {len(train_loader)}")

config = {
    "base_model": "msmarco-bert-base-dot-v5",
    "input_dim": model.get_sentence_embedding_dimension(),
    "output_dim": model.get_sentence_embedding_dimension(),
    "hidden_dim": 784,
    "temperature": 1.0,  # less saturation for tanh
    "loss_temperature": 0.05,  # stronger InfoNCE gradients
    "batch_size": 256,
    "learning_rate_mlp": 0.001,
    "learning_rate_base": 2e-5,
    "warmup_steps": 500,
    "num_epochs": 5,
    "optimizer": "Adam",
    "loss": "InfoNCE",
    "training_mode": "full_scale",  # Training both BERT and MLP
    "use_layernorm": False,  # Set to True to use LayerNorm (may not be necessary)
}

# Initialize wandb
wandb.init(
    project="quantized-mlp-training",
    config=config,
    name=f"full-scale-temp{config['temperature']}-hidden{config['hidden_dim']}"
)

quantized_model = onebitMLP(config["input_dim"], config["output_dim"], config["hidden_dim"], 
                            temperature=config["temperature"], use_layernorm=config.get("use_layernorm", False)).to(device)

model.train()  # Set base model to training mode
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(
    [
        {"params": model.parameters(), "lr": config["learning_rate_base"]},
        {"params": quantized_model.parameters(), "lr": config["learning_rate_mlp"]},
    ]
)

def lr_lambda_base(step: int) -> float:
    if config["warmup_steps"] <= 0:
        return 1.0
    return min(1.0, step / float(config["warmup_steps"]))

def lr_lambda_mlp(step: int) -> float:
    return 1.0

scheduler = LambdaLR(optimizer, lr_lambda=[lr_lambda_base, lr_lambda_mlp])

wandb.config.update({
    "base_model_params": sum(p.numel() for p in model.parameters()),
    "base_model_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "mlp_params": sum(p.numel() for p in quantized_model.parameters()),
    "mlp_trainable": sum(p.numel() for p in quantized_model.parameters() if p.requires_grad),
})

if torch.cuda.is_available():
    base_model_mb = sum(p.numel() * 4 for p in model.parameters()) / 1024**2
    mlp_mb = sum(p.numel() * 4 for p in quantized_model.parameters()) / 1024**2
    current_gb = torch.cuda.memory_allocated() / 1024**3
    max_gb = torch.cuda.max_memory_allocated() / 1024**3
    available_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
    
    print(f"\nGPU Memory Info:")
    print(f"  Base model (trainable): ~{base_model_mb:.1f} MB")
    print(f"  Quantized MLP: ~{mlp_mb:.1f} MB")
    print(f"  Current GPU memory: {current_gb:.2f} GB")
    print(f"  Max GPU memory: {max_gb:.2f} GB")
    print(f"  Available GPU memory: {available_gb:.2f} GB\n")
    
    # Log to wandb
    wandb.log({
        "memory/base_model_mb": base_model_mb,
        "memory/mlp_mb": mlp_mb,
        "memory/current_gb": current_gb,
        "memory/available_gb": available_gb,
    })

def infoNCE_loss(similarity_matrix, temperature=1.0):
    batch_size = similarity_matrix.size(0)
    logits = similarity_matrix / temperature
    labels = torch.arange(batch_size, device=logits.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


model.train()  
quantized_model.train() 
global_step = 0

for epoch in range(config["num_epochs"]):
    epoch_losses = []
    
    for batch_idx, batch in enumerate(train_loader):
        queries = [example.texts[0] for example in batch]
        documents = batch[0].texts[1:]
        query_features = model.tokenize(queries)
        query_features = {k: v.to(device) for k, v in query_features.items()}
        query_embeddings = model(query_features)['sentence_embedding']
        
        doc_features = model.tokenize(documents)
        doc_features = {k: v.to(device) for k, v in doc_features.items()}
        document_embeddings = model(doc_features)['sentence_embedding']
        
        query_vecs = quantized_model(query_embeddings)  # Already L2 normalized in forward()
        doc_vecs = quantized_model(document_embeddings)  # Already L2 normalized in forward()
        similarity_matrix = query_vecs @ doc_vecs.T
        loss = infoNCE_loss(similarity_matrix, temperature=config["loss_temperature"])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        epoch_losses.append(loss.item())
        global_step += 1
        wandb.log({
                "train/batch_loss": loss.item(),
                "train/epoch": epoch + 1,
                "train/global_step": global_step,
            })
    
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"Epoch {epoch+1} loss: {avg_loss:.4f}")
    
    log_dict = {
        "train/epoch_loss": avg_loss,
        "train/epoch": epoch + 1,
    }
    
    # Log GPU memory after each epoch
    if torch.cuda.is_available():
        log_dict.update({
            "memory/current_gb": torch.cuda.memory_allocated() / 1024**3,
            "memory/max_gb": torch.cuda.max_memory_allocated() / 1024**3,
        })
    
    wandb.log(log_dict)

print(f"Training completed. Final epoch loss: {avg_loss:.4f}")

# Save both models
torch.save({
    'base_model_state_dict': model.state_dict(),
    'quantized_model_state_dict': quantized_model.state_dict(),
}, "full_scale_quantized_models.pth")

# Evaluation
dataset = create_eval_dataset(max_queries=100000, use_scoreddocs=False)
test_loader = DataLoader(dataset, batch_size=16384, shuffle=False, collate_fn=in_batch_negatives_collate_fn)

results_quantized = {"correct": 0, "total": 0}
results_non_quantized = {"correct": 0, "total": 0}
model.eval()
quantized_model.eval()
with torch.no_grad():
    for batch in test_loader:
        queries = [example.texts[0] for example in batch]
        documents = batch[0].texts[1:]

        query_embeddings = model.encode(queries, convert_to_tensor=True, device=device)
        document_embeddings = model.encode(documents, convert_to_tensor=True, device=device)
        
        query_vecs_raw = quantized_model(query_embeddings)
        doc_vecs_raw = quantized_model(document_embeddings)
        
        similarity_matrix_non_quant = query_vecs_raw @ doc_vecs_raw.T
        for i in range(len(batch)):
            best = torch.argmax(similarity_matrix_non_quant[i, :])
            if best == i:
                results_non_quantized["correct"] += 1
            results_non_quantized["total"] += 1
        
        query_vecs_quant = one_bit_quantization(query_vecs_raw)
        doc_vecs_quant = one_bit_quantization(doc_vecs_raw)
        similarity_matrix_quant = query_vecs_quant @ doc_vecs_quant.T
        for i in range(len(batch)):
            best = torch.argmax(similarity_matrix_quant[i, :])
            if best == i:
                results_quantized["correct"] += 1
            results_quantized["total"] += 1

accuracy_quantized = results_quantized['correct'] / results_quantized['total']
accuracy_non_quantized = results_non_quantized['correct'] / results_non_quantized['total']
print(f"Accuracy (non-quantized): {accuracy_non_quantized:.5f}")
print(f"Accuracy (1-bit quantized): {accuracy_quantized:.5f}")

wandb.log({
    "eval/accuracy_non_quantized": accuracy_non_quantized,
    "eval/accuracy_quantized": accuracy_quantized,
    "eval/correct_quantized": results_quantized["correct"],
    "eval/total": results_quantized["total"],
})

# Finish wandb run
wandb.finish()

