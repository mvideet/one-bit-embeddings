import torch
from sentence_transformers import SentenceTransformer
from transformer_onebit_ste import IdentityOneBitSTE, one_bit_quantization
from .dataset import create_eval_dataset, in_batch_negatives_collate_fn
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5').to(device)
embed_dim = model.get_sentence_embedding_dimension()


config = {
    "base_model": "msmarco-bert-base-dot-v5",
    "in_dim": embed_dim,
    "use_ste": False,  # Use tanh instead of STE
    "proj_temperature": 1.0,
    "loss_temperature": 0.05,
    "batch_size": 4096,
    "eval_batch_size": 55578,
    "learning_rate": 1e-4,  # Lower LR for stability
    "warmup_steps": 2000,   # Warmup steps
    "num_epochs": 20,
    "optimizer": "Adam",
    "tokens": 8,
    "d_model": 1024,
    "residual_scale": 1e-5,
    "grad_clip": 1.0,        # Gradient clipping norm
    "seed": 42,
}


quantizer = IdentityOneBitSTE(
    in_dim=config["in_dim"],
    use_ste=config["use_ste"],  # False - use tanh
    temperature=config["proj_temperature"],
    tokens=config["tokens"],
    d_model=config["d_model"],
    residual_scale=config["residual_scale"],
    noise_std=config.get("noise_std", 1e-3),
).to(device)

quantizer.load_state_dict(torch.load("./models/transformer_onebit_tanh.pth", map_location=device))
# breakpoint()
print("Num Paramters", sum(p.numel() for p in quantizer.parameters() if p.requires_grad))
# breakpoint()
eval_dataset = create_eval_dataset(max_queries=None, use_scoreddocs=False)
def binarization_error(x):
    return ((x.abs()-1) ** 2).mean().item()
for eval_bs in [2048]:
    config["eval_batch_size"] = eval_bs
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        collate_fn=in_batch_negatives_collate_fn
    )
    print(f"Eval batch size: {eval_bs} — number of batches: {len(eval_loader)}")

    results_quantized = {"correct": 0, "total": 0}
    results_baseline = {"correct": 0, "total": 0}
    quantizer.eval()
    with torch.no_grad():
        for batch in eval_loader:
            queries = [example.texts[0] for example in batch]
            documents = batch[0].texts[1:]

            query_embeddings = model.encode(queries, convert_to_tensor=True, device=device)
            document_embeddings = model.encode(documents, convert_to_tensor=True, device=device)
            q_raw = quantizer(query_embeddings)
            d_raw = quantizer(document_embeddings)
            # breakpoint()
            q_bin = one_bit_quantization(q_raw)
            d_bin = one_bit_quantization(d_raw)

            direct_q = one_bit_quantization(query_embeddings)
            direct_d = one_bit_quantization(document_embeddings)
            sim_direct = direct_q @ direct_d.T
            for i in range(len(batch)):
                best = torch.argmax(sim_direct[i, :])
                if best == i:
                    results_baseline["correct"] += 1
                results_baseline["total"] += 1

            sim_learned = q_bin @ d_bin.T
            for i in range(len(batch)):
                best = torch.argmax(sim_learned[i, :])
                if best == i:
                    results_quantized["correct"] += 1
                results_quantized["total"] += 1

    acc_base = results_baseline['correct'] / max(1, results_baseline['total'])
    acc_learned = results_quantized['correct'] / max(1, results_quantized['total'])
    print(f"[BS={eval_bs}] Eval — Baseline (direct 1-bit): {acc_base:.5f}")
    print(f"[BS={eval_bs}] Eval — Learned (transformer+tanh 1-bit): {acc_learned:.5f}")