import os
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sentence_transformers import SentenceTransformer
from dataset import create_eval_dataset, in_batch_negatives_collate_fn, create_train_dataset
from torch.utils.data import DataLoader
 

class IdentitiyGroupwiseMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_groups, temperature ):
        super(IdentitiyGroupwiseMLP, self).__init__()
        assert input_dim % num_groups == 0
        self.temperature = temperature
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.group_dim = input_dim // num_groups

        layers = []
        for _ in range(num_groups):
            mlp = nn.Sequential(
                nn.Linear(self.group_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.group_dim)
            )
            layers.append(mlp)
        self.group_mlps = nn.ModuleList(layers)
        self._init_small_noise(0.001)
        
    def _init_small_noise(self, noise_std: float) -> None:
        with torch.no_grad():
            for mlp in self.group_mlps:
                for name, param in mlp.named_parameters():
                    if 'weight' in name:
                        nn.init.normal_(param, mean=0.0, std=0.001)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        groups = torch.split(x, self.group_dim, dim=1)
        outs = []
        for group_x, mlp in zip(groups, self.group_mlps):
            out = mlp(group_x)
            outs.append(out)
        out = torch.cat(outs, dim=1)
        out = out + x
        out_tanh = torch.tanh(out / self.temperature)
        out_sign = torch.sign(out_tanh)
        # Forward uses sign, backward uses tanh gradient
        out_ste = out_tanh + (out_sign - out_tanh).detach()
        return out_ste
    

def one_bit_quantization(x):
    return torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))

def infoNCE_loss(similarity_matrix: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    batch_size = similarity_matrix.size(0)
    logits = similarity_matrix / temperature
    labels = torch.arange(batch_size, device=logits.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')
    model = model.to(device)
    print(model)

    train_dataset = create_train_dataset(use_scoreddocs=False)
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, collate_fn=in_batch_negatives_collate_fn)
    print("loading data done")
    print(f"number of batches in training: {len(train_loader)}")
    
    config = {
        "input_dim": model.get_sentence_embedding_dimension(),
        "output_dim": model.get_sentence_embedding_dimension(),
        "hidden_dim": 1024,
        "num_groups": 1,
        "temperature": 1.0,
        "learning_rate": 1e-4,
        "num_epochs": 20,
        "eval_batch_size": 16384,
        "loss_temperature": 0.05,
        "grad_clip": 1.0,
    }
    
    wandb.init(project="groupwise-mlp", name="groupwise-mlp")
    wandb.config.update(config)
    
    quantizer = IdentitiyGroupwiseMLP(input_dim=config["input_dim"], output_dim=config["output_dim"], hidden_dim=config["hidden_dim"], num_groups=config["num_groups"], temperature=config["temperature"]).to(device)
    optimizer = torch.optim.Adam(quantizer.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    eval_dataset = create_eval_dataset(max_queries=None, use_scoreddocs=False)
    eval_loader = DataLoader(eval_dataset, batch_size=config["eval_batch_size"], shuffle=False, collate_fn=in_batch_negatives_collate_fn)
    print(f"number of batches in eval: {len(eval_loader)}")

    # Initial evaluation on full dev set (epoch 0)
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
    print(f"Epoch 0 eval — Learned (transformer+tanh 1-bit): {acc_learned:.5f}")
    wandb.log({
        "eval/accuracy_baseline_onebit": acc_base,
        "eval/accuracy_learned_onebit": acc_learned,
        "eval/epoch": 0,
    })
    quantizer.train()

    # Training
    global_step = 0
    best_dev_acc = float("-inf")
    for epoch in range(config["num_epochs"]):
        quantizer.train()
        epoch_losses = []  # total loss
        epoch_contrastive_losses = []
        
        for batch in train_loader:
            queries = [example.texts[0] for example in batch]
            documents = batch[0].texts[1:]

            with torch.no_grad():
                query_embeddings = model.encode(queries, convert_to_tensor=True, device=device)
                document_embeddings = model.encode(documents, convert_to_tensor=True, device=device)

            query_embeddings = query_embeddings.clone()
            document_embeddings = document_embeddings.clone()

            # Forward
            qz = quantizer(query_embeddings)
            dz = quantizer(document_embeddings)
            sim = qz @ dz.T
            contrastive_loss = infoNCE_loss(sim, temperature=config["loss_temperature"])
            loss = contrastive_loss
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(quantizer.parameters(), config.get("grad_clip", 1.0))
            
            optimizer.step()
            
            # Collect stats
            epoch_losses.append(loss.item())
            epoch_contrastive_losses.append(contrastive_loss.item())
            
            global_step += 1
            
            if len(epoch_losses) % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                wandb.log({
                    "train/batch_total_loss": loss.item(),
                    "train/batch_contrastive_loss": contrastive_loss.item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/learning_rate": current_lr,
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step,
                })
        
        avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        avg_contrastive = sum(epoch_contrastive_losses) / max(1, len(epoch_contrastive_losses))
        
        # Step scheduler once per epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1} total_loss: {avg_loss:.4f}, contrastive: {avg_contrastive:.4f}, LR: {current_lr:.6f}")
        wandb.log({
            "train/epoch_total_loss": avg_loss,
            "train/epoch_contrastive_loss": avg_contrastive,
            "train/epoch": epoch + 1,
            "train/learning_rate": current_lr,
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
        print(f"Epoch {epoch+1} eval — Baseline (direct 1-bit): {acc_base:.5f}")
        print(f"Epoch {epoch+1} eval — Learned (transformer+tanh 1-bit): {acc_learned:.5f}")
        wandb.log({
            "eval/accuracy_baseline_onebit": acc_base,
            "eval/accuracy_learned_onebit": acc_learned,
            "eval/epoch": epoch + 1,
        })
        if acc_learned > best_dev_acc:
            best_dev_acc = acc_learned
            torch.save(quantizer.state_dict(), "groupwise_mlp_best.pth")
            wandb.log({
                "eval/best_accuracy_learned_onebit": best_dev_acc,
                "checkpoint/saved": 1,
            })
        quantizer.train()

    print(f"Training completed. Final epoch loss: {avg_loss:.4f}")

    torch.save(quantizer.state_dict(), "groupwise_mlp_final.pth")

    wandb.finish()


