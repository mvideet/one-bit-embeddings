import torch
import torch.nn as nn
import os
os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER', None)
import wandb
import hashlib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dataset import create_eval_dataset, in_batch_negatives_collate_fn, create_train_dataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')
model = model.to(device)
print(model)


class ResidualMLPOneBitSTE(nn.Module):
    """
    Two-layer MLP with GELU and residual:
      y = x + residual_scale * W2(GELU(W1 x))
    Then L2 normalize and binarize with STE (or tanh when use_ste=False).
    Small weight init keeps behavior near the baseline initially.
    """
    def __init__(self, in_dim: int, hidden_dim: int, use_ste: bool = True, temperature: float = 1.0, residual_scale: float = 1e-2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim, bias=True)
        self._init_small_noise()
        self.use_ste = use_ste
        self.temperature = temperature
        self.residual_scale = residual_scale

    def _init_small_noise(self) -> None:
        with torch.no_grad():
            noise_std = 1e-3
            nn.init.normal_(self.fc1.weight, mean=0.0, std=noise_std)
            nn.init.normal_(self.fc2.weight, mean=0.0, std=noise_std)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mlp_out = self.fc2(self.act(self.fc1(x)))
        z = x + self.residual_scale * mlp_out  # near-identity at init
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
    "hidden_dim": 1024,
    "use_ste": True,
    "proj_temperature": 1.0,
    "loss_temperature": 0.05,
    "residual_scale": 1e-2,
    "batch_size": 4096,
    "learning_rate": 5e-4,
    "num_epochs": 5,
    "optimizer": "Adam",
    "loss": "InfoNCE",
    "quant_loss_weight": 0.01,
    "use_embedding_cache": True,
    "embedding_cache_dir": "cache/encodings",
}

# Initialize wandb
wandb.init(
    project="quantized-mlp-training",
    config=config,
    name="mlp-onebit-ste"
)

quantizer = ResidualMLPOneBitSTE(
    in_dim=config["in_dim"],
    hidden_dim=config["hidden_dim"],
    use_ste=config["use_ste"],
    temperature=config["proj_temperature"],
    residual_scale=config["residual_scale"],
).to(device)
optimizer = torch.optim.Adam(quantizer.parameters(), lr=config["learning_rate"])

# Log model architecture
wandb.config.update({
    "quantizer_params": sum(p.numel() for p in quantizer.parameters()),
    "quantizer_trainable": sum(p.numel() for p in quantizer.parameters() if p.requires_grad),
    "quant_loss_weight": config["quant_loss_weight"],
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


def _sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def _get_cache_path_for_model(model_name: str, cache_dir: str) -> str:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return os.path.join(cache_dir, f"embeddings_{_sanitize_model_name(model_name)}.pth")


def _encode_texts_with_cache(texts, model_obj, device_obj, cache_path: str) -> torch.Tensor:
    """
    Encode a list of texts using a persistent on-disk cache (.pth).
    Caches per-text embeddings on CPU; returns a stacked tensor on the requested device.
    """
    if not config.get("use_embedding_cache", True):
        return model_obj.encode(texts, convert_to_tensor=True, device=device_obj)

    if os.path.exists(cache_path):
        text_to_vec = torch.load(cache_path, map_location="cpu")
    else:
        text_to_vec = {}

    # Pre-compute keys for all texts
    keys = []
    for t in texts:
        h = hashlib.sha1()
        h.update(t.encode("utf-8"))
        keys.append(h.hexdigest())

    # Identify which embeddings are missing from cache
    missing_indices = [i for i, k in enumerate(keys) if k not in text_to_vec]
    if missing_indices:
        to_encode = [texts[i] for i in missing_indices]
        new_vecs = model_obj.encode(to_encode, convert_to_tensor=True, device=device_obj)
        if new_vecs.ndim == 1:
            new_vecs = new_vecs.unsqueeze(0)
        # Store on CPU to keep cache device-agnostic
        for i, vec in zip(missing_indices, new_vecs):
            text_to_vec[keys[i]] = vec.detach().cpu()
        torch.save(text_to_vec, cache_path)

    # Assemble in original order and move to target device
    stacked = torch.stack([text_to_vec[k] for k in keys], dim=0).to(device_obj)
    return stacked


def _dump_embeddings_for_loader(loader: DataLoader, split_name: str) -> None:
    """
    Iterate over a dataloader, encode queries and documents using the cache,
    and save aggregated 3D tensors on CPU:
      - queries:  [num_batches, batch_size, embed_dim]
      - documents:[num_batches, batch_size, embed_dim]
    The final batch is padded with zeros if it is smaller than batch_size.
    Also saves per-batch valid lengths.
    """
    num_batches = len(loader)
    bsz = loader.batch_size or config["batch_size"]
    edim = config["in_dim"]
    cache_path = _get_cache_path_for_model(config["base_model"], config["embedding_cache_dir"])

    queries_all = torch.zeros((num_batches, bsz, edim), dtype=torch.float32)
    documents_all = torch.zeros((num_batches, bsz, edim), dtype=torch.float32)
    lengths = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            queries = [example.texts[0] for example in batch]
            documents = batch[0].texts[1:]

            q_emb = _encode_texts_with_cache(queries, model, device, cache_path).to("cpu", dtype=torch.float32)
            d_emb = _encode_texts_with_cache(documents, model, device, cache_path).to("cpu", dtype=torch.float32)

            n = q_emb.shape[0]
            lengths.append(int(n))

            if n > bsz:
                q_emb = q_emb[:bsz]
                d_emb = d_emb[:bsz]
                n = bsz

            queries_all[batch_idx, :n, :] = q_emb
            documents_all[batch_idx, :n, :] = d_emb

    out_dir = Path(config["embedding_cache_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "split": split_name,
        "num_batches": num_batches,
        "batch_size": bsz,
        "embed_dim": edim,
        "lengths": lengths,
        "base_model": config["base_model"],
    }
    torch.save({"embeddings": queries_all, "meta": meta}, os.path.join(out_dir, f"{split_name}_query_embeddings.pth"))
    torch.save({"embeddings": documents_all, "meta": meta}, os.path.join(out_dir, f"{split_name}_document_embeddings.pth"))
    print(f"Saved {split_name} query/document embeddings to {out_dir}")


# Training loop
quantizer.train()
global_step = 0

with torch.no_grad():
    tmp_sentences = ["hello world", "this is a test", "another query"]
    tmp_q = model.encode(tmp_sentences, convert_to_tensor=True, device=device)
    base_bin = one_bit_quantization(tmp_q)
    learned = one_bit_quantization(quantizer(tmp_q))
    exact_match = torch.all((base_bin == learned))
    print(f"Sanity check - exact baseline match at init: {bool(exact_match)}")

# Initial evaluation on full dev set (epoch 0)
results_quantized = {"correct": 0, "total": 0}
results_baseline = {"correct": 0, "total": 0}
quantizer.eval()
with torch.no_grad():
    for batch in eval_loader:
        queries = [example.texts[0] for example in batch]
        documents = batch[0].texts[1:]

        _cache_path = _get_cache_path_for_model(config["base_model"], config["embedding_cache_dir"])
        query_embeddings = _encode_texts_with_cache(queries, model, device, _cache_path)  # (batch_size, embed_dim)
        document_embeddings = _encode_texts_with_cache(documents, model, device, _cache_path)  # (batch_size, embed_dim)

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
print(f"Epoch 0 eval — Learned (identity+STE 1-bit): {acc_learned:.5f}")
wandb.log({
    "eval/accuracy_baseline_onebit": acc_base,
    "eval/accuracy_learned_onebit": acc_learned,
    "eval/epoch": 0,
})
quantizer.train()

# for epoch in range(config["num_epochs"]):
#     epoch_total_losses = []
#     epoch_contrastive_losses = []
#     epoch_quant_losses = []
#
#     for batch in train_loader:
#         queries = [example.texts[0] for example in batch]
#         documents = batch[0].texts[1:]
#
#         with torch.no_grad():
#             _cache_path = _get_cache_path_for_model(config["base_model"], config["embedding_cache_dir"])
#             query_embeddings = _encode_texts_with_cache(queries, model, device, _cache_path)
#             document_embeddings = _encode_texts_with_cache(documents, model, device, _cache_path)
#
#         query_embeddings = query_embeddings.clone()
#         document_embeddings = document_embeddings.clone()
#
#         query_vecs = quantizer(query_embeddings)
#         doc_vecs = quantizer(document_embeddings)
#
#         sim = query_vecs @ doc_vecs.T
#         contrastive_loss = infoNCE_loss(sim, temperature=config["loss_temperature"])
#         # Quantization loss to encourage values to +/-1
#         quant_loss_q = ((query_vecs.abs() - 1) ** 2).mean()
#         quant_loss_d = ((doc_vecs.abs() - 1) ** 2).mean()
#         quant_loss = 0.5 * (quant_loss_q + quant_loss_d)
#         loss = contrastive_loss + config["quant_loss_weight"] * quant_loss
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         epoch_total_losses.append(loss.item())
#         epoch_contrastive_losses.append(contrastive_loss.item())
#         epoch_quant_losses.append(quant_loss.item())
#         global_step += 1
#         if global_step % 10 == 0:
#             wandb.log({
#                 "train/batch_total_loss": loss.item(),
#                 "train/batch_contrastive_loss": contrastive_loss.item(),
#                 "train/batch_quant_loss": quant_loss.item(),
#                 "train/epoch": epoch + 1,
#                 "train/global_step": global_step,
#             })
#
#     avg_total = sum(epoch_total_losses) / max(1, len(epoch_total_losses))
#     avg_contrastive = sum(epoch_contrastive_losses) / max(1, len(epoch_contrastive_losses))
#     avg_quant = sum(epoch_quant_losses) / max(1, len(epoch_quant_losses))
#     print(f"Epoch {epoch+1} total_loss: {avg_total:.4f}, contrastive: {avg_contrastive:.4f}, quant: {avg_quant:.4f}")
#     wandb.log({
#         "train/epoch_total_loss": avg_total,
#         "train/epoch_contrastive_loss": avg_contrastive,
#         "train/epoch_quant_loss": avg_quant,
#         "train/epoch": epoch + 1,
#     })
#
#     # Per-epoch evaluation on full dev set
#     results_quantized = {"correct": 0, "total": 0}
#     results_baseline = {"correct": 0, "total": 0}
#     quantizer.eval()
#     with torch.no_grad():
#         for batch in eval_loader:
#             queries = [example.texts[0] for example in batch]
#             documents = batch[0].texts[1:]
#
#             _cache_path = _get_cache_path_for_model(config["base_model"], config["embedding_cache_dir"])
#             query_embeddings = _encode_texts_with_cache(queries, model, device, _cache_path)
#             document_embeddings = _encode_texts_with_cache(documents, model, device, _cache_path)
#
#             # Baseline: direct 1-bit from base embeddings
#             baseline_q = one_bit_quantization(query_embeddings)
#             baseline_d = one_bit_quantization(document_embeddings)
#             sim_base = baseline_q @ baseline_d.T
#             for i in range(len(batch)):
#                 best = torch.argmax(sim_base[i, :])
#                 if best == i:
#                     results_baseline["correct"] += 1
#                 results_baseline["total"] += 1
#
#             # Learned 1-bit: identity + STE 1-bit
#             q_raw = quantizer(query_embeddings)
#             d_raw = quantizer(document_embeddings)
#             q_bin = one_bit_quantization(q_raw)
#             d_bin = one_bit_quantization(d_raw)
#             sim_learned = q_bin @ d_bin.T
#             for i in range(len(batch)):
#                 best = torch.argmax(sim_learned[i, :])
#                 if best == i:
#                     results_quantized["correct"] += 1
#                 results_quantized["total"] += 1
#
#     acc_base = results_baseline['correct'] / max(1, results_baseline['total'])
#     acc_learned = results_quantized['correct'] / max(1, results_quantized['total'])
#     print(f"Epoch {epoch+1} eval — Baseline (direct 1-bit): {acc_base:.5f}")
#     print(f"Epoch {epoch+1} eval — Learned (identity+STE 1-bit): {acc_learned:.5f}")
#     wandb.log({
#         "eval/accuracy_baseline_onebit": acc_base,
#         "eval/accuracy_learned_onebit": acc_learned,
#         "eval/epoch": epoch + 1,
#     })
#     quantizer.train()

# print(f"Training completed. Final epoch loss: {avg_loss:.4f}")
#
# torch.save(quantizer.state_dict(), "identity_onebit_ste.pth")
#
# wandb.finish()

# Dump full-dataset embeddings for train and eval splits (cached encodings reused)
print("Dumping full-dataset embeddings (queries and documents) with caching...")
train_dump_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=in_batch_negatives_collate_fn)
_dump_embeddings_for_loader(train_dump_loader, "train")
eval_dump_loader = DataLoader(eval_dataset, batch_size=eval_loader.batch_size, shuffle=False, collate_fn=in_batch_negatives_collate_fn)
_dump_embeddings_for_loader(eval_dump_loader, "eval")
print("Done.")

