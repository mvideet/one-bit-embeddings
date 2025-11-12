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
 


class IdentityOneBitSTE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        use_ste: bool = True,
        temperature: float = 1.0,
        tokens: int = 4,
        d_model: int = None,
        nhead: int = 12,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        residual_scale: float = 1e-2,
        noise_std: float = 1e-3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = in_dim if d_model is None else d_model
        self.tokens = tokens
        self.use_ste = use_ste
        self.temperature = temperature
        self.residual_scale = nn.Parameter(torch.zeros(1, self.in_dim))
        self.in_proj = nn.Linear(self.in_dim, self.tokens * self.d_model, bias=True)
        self.pos = nn.Parameter(torch.zeros(1, self.tokens, self.d_model))
        self.nhead = nhead
        if self.d_model % self.nhead != 0:
            target_head_dim = 64
            candidate = max(1, self.d_model // target_head_dim)
            while candidate > 1 and (self.d_model % candidate != 0):
                candidate -= 1
            self.nhead = candidate if (self.d_model % candidate == 0) else 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(self.d_model, self.in_dim, bias=True)

        self._init_small_noise(noise_std)
    
    def _init_small_noise(self, noise_std: float) -> None:
        with torch.no_grad():
            nn.init.normal_(self.in_proj.weight, mean=0.0, std=noise_std)
            nn.init.zeros_(self.in_proj.bias)
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
            nn.init.normal_(self.pos, mean=0.0, std=noise_std)
            for layer in self.encoder.layers:
                for name, param in layer.named_parameters():
                    if 'weight' in name:
                        if 'norm' in name:
                            continue
                        nn.init.normal_(param, mean=0.0, std=noise_std)
                    elif 'bias' in name:
                        if 'norm' in name:
                            continue
                        nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, return_stats: bool = False) -> torch.Tensor:
        B = x.size(0)
        tokens = self.in_proj(x).view(B, self.tokens, self.d_model) + self.pos 
        h = self.encoder(tokens) 
        pooled = h.mean(dim=1)           
        delta = self.out_proj(pooled) 
        raw_scale = self.residual_scale
        scale = 0.1 * torch.tanh(raw_scale) + 0.1
        scale = scale.expand(B, -1)         
        z = x + scale * delta
        z = torch.nn.functional.normalize(z, p=2, dim=1)
        if self.use_ste:
            s = torch.sign(z)
            q = z + (s - z).detach()
        else:
            z_tanh = torch.tanh(z / self.temperature)
            z_sign = torch.sign(z_tanh)
            q = z_tanh + (z_sign - z_tanh).detach()
        if return_stats:
            return q, delta, scale
        return q


def one_bit_quantization(embeddings: torch.Tensor) -> torch.Tensor:
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return torch.where(embeddings > 0, torch.ones_like(embeddings), -torch.ones_like(embeddings))


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
        "use_ste": False,  # Use tanh instead of STE
        "proj_temperature": 0.75,
        "loss_temperature": 0.05,
        "batch_size": 4096,
        "eval_batch_size": 16384,
        "learning_rate": 5e-4,  # Lower LR for stability
        "warmup_steps": 1000,   # Warmup steps
        "num_epochs": 20,
        "optimizer": "Adam",
        "tokens": 8,
        "d_model": 1024,
        "residual_scale": 1e-4,
        "grad_clip": 1.0,        # Gradient clipping norm
        "seed": 42,
        "quant_loss_weight": 0.00,
    }

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # Initialize wandb
    wandb.init(
        project="quantized-mlp-training",
        config=config,
        name="transformer-onebit-tanh"
    )

    # Model - use tanh only
    quantizer = IdentityOneBitSTE(
        in_dim=config["in_dim"],
        use_ste=config["use_ste"],  # False - use tanh
        temperature=config["proj_temperature"],
        tokens=config["tokens"],
        d_model=config["d_model"],
        residual_scale=config["residual_scale"],
        noise_std=config.get("noise_std", 1e-3),
    ).to(device)
    optimizer = torch.optim.Adam(quantizer.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = config.get("warmup_steps", 2000)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Log model stats
    wandb.config.update({
        "in_dim": config["in_dim"],
        "tokens": config["tokens"],
        "d_model": config["d_model"],
        "nhead": getattr(quantizer, "nhead", None),
        "params": sum(p.numel() for p in quantizer.parameters()),
        "trainable": sum(p.numel() for p in quantizer.parameters() if p.requires_grad),
        "quant_loss_weight": config["quant_loss_weight"],
    })

    # Create eval dataset and loader
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
        epoch_quant_losses = []
        epoch_delta_norms = []
        epoch_scale_means = []
        epoch_scale_maxs = []
        
        for batch in train_loader:
            queries = [example.texts[0] for example in batch]
            documents = batch[0].texts[1:]

            with torch.no_grad():
                query_embeddings = model.encode(queries, convert_to_tensor=True, device=device)
                document_embeddings = model.encode(documents, convert_to_tensor=True, device=device)

            query_embeddings = query_embeddings.clone()
            document_embeddings = document_embeddings.clone()

            # Forward with stats
            qz, q_delta, q_scale = quantizer(query_embeddings, return_stats=True)
            dz, d_delta, d_scale = quantizer(document_embeddings, return_stats=True)
            sim = qz @ dz.T
            contrastive_loss = infoNCE_loss(sim, temperature=config["loss_temperature"])
            # Quantization regularizer to push outputs toward +/-1
            quant_loss_q = ((qz.abs() - 1) ** 2).mean()
            quant_loss_d = ((dz.abs() - 1) ** 2).mean()
            # quant_loss = 0.5 * (quant_loss_q + quant_loss_d)
            # loss = contrastive_loss + config["quant_loss_weight"] * quant_loss
            loss = contrastive_loss
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(quantizer.parameters(), config.get("grad_clip", 1.0))
            
            optimizer.step()
            scheduler.step()
            
            # Collect stats
            epoch_losses.append(loss.item())
            epoch_contrastive_losses.append(contrastive_loss.item())
            # epoch_quant_losses.append(quant_loss.item())
            epoch_delta_norms.append((q_delta.abs().mean().item() + d_delta.abs().mean().item()) / 2)
            epoch_scale_means.append((q_scale.mean().item() + d_scale.mean().item()) / 2)
            epoch_scale_maxs.append((q_scale.max().item() + d_scale.max().item()) / 2)
            
            global_step += 1
            
            if len(epoch_losses) % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                wandb.log({
                    "train/batch_total_loss": loss.item(),
                    "train/batch_contrastive_loss": contrastive_loss.item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/learning_rate": current_lr,
                    "train/delta_norm": epoch_delta_norms[-1],
                    "train/scale_mean": epoch_scale_means[-1],
                    "train/scale_max": epoch_scale_maxs[-1],
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step,
                })
        
        avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        avg_delta_norm = sum(epoch_delta_norms) / max(1, len(epoch_delta_norms))
        avg_scale_mean = sum(epoch_scale_means) / max(1, len(epoch_scale_means))
        avg_scale_max = sum(epoch_scale_maxs) / max(1, len(epoch_scale_maxs))
        current_lr = scheduler.get_last_lr()[0]
        
        avg_contrastive = sum(epoch_contrastive_losses) / max(1, len(epoch_contrastive_losses))
        print(f"Epoch {epoch+1} total_loss: {avg_loss:.4f}, contrastive: {avg_contrastive:.4f}, LR: {current_lr:.6f}, delta_norm: {avg_delta_norm:.6f}, scale_mean: {avg_scale_mean:.6f}, scale_max: {avg_scale_max:.6f}")
        wandb.log({
            "train/epoch_total_loss": avg_loss,
            "train/epoch_contrastive_loss": avg_contrastive,
            "train/epoch": epoch + 1,
            "train/learning_rate": current_lr,
            "train/delta_norm": avg_delta_norm,
            "train/scale_mean": avg_scale_mean,
            "train/scale_max": avg_scale_max,
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
        # Save best on dev
        if acc_learned > best_dev_acc:
            best_dev_acc = acc_learned
            torch.save(quantizer.state_dict(), "quantize_loss_transformer.pth")
            wandb.log({
                "eval/best_accuracy_learned_onebit": best_dev_acc,
                "checkpoint/saved": 1,
            })
        quantizer.train()
    torch.save(quantizer.state_dict(), "transformer_onebit_tanh_run2.pth")

    wandb.finish()
