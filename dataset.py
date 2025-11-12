import ir_datasets
import random
from torch.utils.data import Dataset
from collections import defaultdict
from typing import Optional, List

try:
    from sentence_transformers import InputExample
except ImportError:
    class InputExample:
        def __init__(self, texts, label=None):
            self.texts = texts
            self.label = label


class MSMARCODatasetIR(Dataset):
    
    def __init__(
        self, 
        dataset_name: str = "msmarco-passage/train",
        max_queries: Optional[int] = None,
        use_scoreddocs: bool = False,
        num_negatives_per_query: int = 1
    ):
        self.dataset_name = dataset_name
        self.num_negatives_per_query = num_negatives_per_query
        self.irds_dataset = ir_datasets.load(dataset_name)
        self._build_mappings(max_queries, use_scoreddocs)
        
        print(f"Dataset ready: {len(self.query_ids)} queries")
    
    def _build_mappings(self, max_queries: Optional[int], use_scoreddocs: bool):
        
        self.doc_store = self.irds_dataset.docs_store()
        self.query_to_positives = defaultdict(list)
        qrel_iter = self.irds_dataset.qrels_iter()
        
        for qrel in qrel_iter:
            if qrel.relevance == 1:
                self.query_to_positives[qrel.query_id].append(qrel.doc_id)
                if max_queries and len(self.query_to_positives) >= max_queries:
                    break
                
        self.queries = {}
        query_iter = self.irds_dataset.queries_iter()
        queries_with_positives = set(self.query_to_positives.keys())
        
        for query in query_iter:
            if query.query_id in queries_with_positives:
                self.queries[query.query_id] = query.text
                if len(self.queries) >= len(queries_with_positives):
                    break
        
        self.query_to_negatives = defaultdict(list)
        
        if use_scoreddocs and hasattr(self.irds_dataset, 'scoreddocs_iter'):
            scoreddocs_iter = self.irds_dataset.scoreddocs_iter()
            for scored_doc in scoreddocs_iter:
                if scored_doc.query_id in self.queries:
                    if scored_doc.doc_id not in self.query_to_positives[scored_doc.query_id]:
                        self.query_to_negatives[scored_doc.query_id].append(scored_doc.doc_id)
                        if len(self.query_to_negatives[scored_doc.query_id]) >= self.num_negatives_per_query * 10:
                            break
        else:
            doc_pool = []
            doc_pool_set = set()
            for doc in self.irds_dataset.docs_iter():
                doc_pool.append(doc.doc_id)
                doc_pool_set.add(doc.doc_id)
                if len(doc_pool) >= 500000:
                    break
                        
            random.shuffle(doc_pool)
            for qid in self.query_to_positives.keys():
                positives_set = set(self.query_to_positives[qid])
                negatives = []
                
                for doc_id in doc_pool:
                    if doc_id not in positives_set:
                        negatives.append(doc_id)
                        if len(negatives) >= self.num_negatives_per_query * 10:
                            break
                
                if len(negatives) < self.num_negatives_per_query:
                    for doc in self.irds_dataset.docs_iter():
                        if doc.doc_id not in positives_set and doc.doc_id not in doc_pool_set:
                            negatives.append(doc.doc_id)
                            if len(negatives) >= self.num_negatives_per_query * 10:
                                break
                
                self.query_to_negatives[qid] = negatives
        
        self.query_ids = []
        for qid in self.queries.keys():
            if len(self.query_to_positives[qid]) > 0 and len(self.query_to_negatives[qid]) > 0:
                self.query_ids.append(qid)
        
        for qid in self.query_ids:
            self.query_to_positives[qid] = list(self.query_to_positives[qid])
            self.query_to_negatives[qid] = list(self.query_to_negatives[qid])
            random.shuffle(self.query_to_positives[qid])
            random.shuffle(self.query_to_negatives[qid])
    
    def __len__(self):
        return len(self.query_ids)
    
    def __getitem__(self, idx):
        qid = self.query_ids[idx]
        query_text = self.queries[qid]
        
        if len(self.query_to_positives[qid]) > 0:
            pos_id = self.query_to_positives[qid].pop(0)
            pos_text = self.doc_store.get(pos_id).text
            self.query_to_positives[qid].append(pos_id)
        else:
            pos_id = self.query_to_negatives[qid].pop(0)
            pos_text = self.doc_store.get(pos_id).text
            self.query_to_negatives[qid].append(pos_id)
        
        return {
            'query': query_text,
            'positive': pos_text,
            'query_id': qid
        }


def in_batch_negatives_collate_fn(batch):
    queries = [item['query'] for item in batch]
    positives = [item['positive'] for item in batch]
    
    all_documents = positives
    
    examples = []
    for i in range(len(batch)):
        query = queries[i]
        
        examples.append(InputExample(
            texts=[query] + all_documents,
            label=i
        ))
    
    return examples


def create_train_dataset(max_queries: Optional[int] = None, use_scoreddocs: bool = False):
    return MSMARCODatasetIR(
        dataset_name="msmarco-passage/train",
        max_queries=max_queries,
        use_scoreddocs=use_scoreddocs
    )


def create_eval_dataset(max_queries: Optional[int] = None, use_scoreddocs: bool = False):
    return MSMARCODatasetIR(
        dataset_name="msmarco-passage/dev",
        max_queries=max_queries,
        use_scoreddocs=use_scoreddocs
    )


if __name__ == "__main__":
    print("Creating training dataset...")
    train_dataset = create_train_dataset(max_queries=100, use_scoreddocs=False)
    
    # print(f"\nDataset size: {len(train_dataset)}")
    # print("\nSample training example:")
    # example = train_dataset[0]
    # print(f"Query: {example['query'][:100]}...")
    # print(f"Positive: {example['positive'][:100]}...")
    
    print("\n" + "="*80)
    print("Testing in-batch negatives collate function...")
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=in_batch_negatives_collate_fn)
    batch = next(iter(train_loader))
    for i, example in enumerate(batch):
        print(f"Example {i}: texts = {example.texts}, label = {example.label}")
