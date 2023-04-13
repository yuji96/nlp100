# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: yuji
#     language: python
#     name: python3
# ---

# +
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = "cuda"


# +
# 80
import pandas as pd

train = pd.read_table("train.tsv")
valid = pd.read_table("valid.tsv")
test = pd.read_table("test.tsv")

# +
from itertools import chain

vocab = chain.from_iterable(train["title"].str.split())
vocab = pd.Series(vocab, name="word")
vocab = vocab.value_counts().reset_index(name="count")
vocab = vocab.loc[vocab["count"] >= 2, "word"]
vocab = {w: i for i, w in enumerate(vocab.tolist(), start=1)}
assert "<unk>" not in vocab
vocab["<unk>"] = 0
len(vocab)

# +
from dataset import Tokenier

tokenizer = Tokenier(vocab)
tokenizer("hello world This is a pen you are kind")

# -

# %load_ext autoreload
# %autoreload 2

len(vocab)

max(vocab.values())

# +
# 81
import torch
from models import RNN

model = RNN(vocab_size=len(vocab), hidden_size=300, output_size=5)
model(torch.tensor([[1, 2, 3]]), labels=torch.tensor([3]))


# +
# 82
# TODO: trainer に移動
from sklearn.metrics import accuracy_score, f1_score
from transformers import EvalPrediction


def custom_compute_metrics(res: EvalPrediction) -> dict:
    pred = res.predictions.argmax(axis=1)
    acc = accuracy_score(res.label_ids, pred)
    f1 = f1_score(res.label_ids, pred, average="macro")
    return {"accuracy": acc, "f1": f1}


from dataset import NewsAggCollator, NewsAggDataset

# +
from transformers import TrainingArguments
from transformers.trainer import Trainer

training_args = TrainingArguments(
    output_dir="/data/yuji96/nlp100/82_rnn_sgd",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=15,
    remove_unused_columns=False,
    report_to="wandb",
)


# +
from transformers import EarlyStoppingCallback

model = model.to(device)
collator = NewsAggCollator(tokenizer, device=device)

sgd = torch.optim.SGD(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.ConstantLR(sgd, factor=1, total_iters=1)
scheduler = None
trainer = Trainer(
    model,
    training_args,
    train_dataset=NewsAggDataset(train),
    eval_dataset=NewsAggDataset(valid),
    data_collator=collator,
    compute_metrics=custom_compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    optimizers=(sgd, scheduler),
)
trainer.train()
