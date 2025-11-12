import torch
import os
os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER', None)
from sentence_transformers import SentenceTransformer
from dataset import create_eval_dataset, in_batch_negatives_collate_fn
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')
model = model.to(device)
print(model)


def one_bit_quantization(embeddings):
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    # -1 if less than 0, 1 if greater than 0
    quantized = torch.where(embeddings > 0, torch.ones_like(embeddings), -torch.ones_like(embeddings))
    return quantized


dataset = create_eval_dataset(max_queries=100000, use_scoreddocs=False)
test_loader = DataLoader(dataset, batch_size= 55578, shuffle=False, collate_fn=in_batch_negatives_collate_fn)
print("loading data done")
print(f"number of batches: {len(test_loader)}")
results = {"quantized": {"correct": 0, "total": 0},
           "not_quantized": {"correct": 0, "total": 0}}

with torch.no_grad():
    for batch in test_loader:
        queries = [example.texts[0] for example in batch]
        documents = batch[0].texts[1:]

        query_embeddings = model.encode(queries, convert_to_tensor=True, device=device)
        document_embeddings = model.encode(documents, convert_to_tensor=True, device=device)

        for use_quant in [False, True]:
            if use_quant:
                query_vecs = one_bit_quantization(query_embeddings).float()
                doc_vecs = one_bit_quantization(document_embeddings).float()
                key = "quantized"
            else:
                query_vecs = query_embeddings
                doc_vecs = document_embeddings
                key = "not_quantized"

            similarity_matrix = query_vecs @ doc_vecs.T

            for i in range(len(batch)):
                best = torch.argmax(similarity_matrix[i, :])
                if best == i:
                    results[key]["correct"] += 1
                results[key]["total"] += 1

print(f"Accuracy without quantization: {results['not_quantized']['correct'] / results['not_quantized']['total']:.5f}")
print(f"Accuracy with quantization: {results['quantized']['correct'] / results['quantized']['total']:.5f}")

