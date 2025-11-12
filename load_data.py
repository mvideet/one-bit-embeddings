import ir_datasets
from dataset import MSMARCODatasetIR, create_train_dataset, create_eval_dataset
from torch.utils.data import DataLoader

# Option 1: Quick exploration of the dataset structure
dataset = ir_datasets.load("msmarco-passage/train")
docs = dataset.docs_iter()
next(docs)

queries_iter = dataset.queries_iter()
next(queries_iter)

qrels_iter = dataset.qrels_iter()
next(qrels_iter)

eval_dataset_irds = ir_datasets.load("msmarco-passage/dev")
eval_docs = eval_dataset_irds.docs_iter()
next(eval_docs)

train_dataset = create_train_dataset(max_queries=1000, use_scoreddocs=False)

eval_dataset = create_eval_dataset(max_queries=100, use_scoreddocs=False)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    drop_last=True
)

eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=8,
    shuffle=False
)

# Option 4: Test a batch
batch = next(iter(train_dataloader))
