import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class Tokenier:
    def __init__(self, vocab: dict) -> None:
        self.vocab = vocab

    def __call__(self, text: str) -> list:
        words = text.split()
        token_ids = [self.vocab.get(w, 0) for w in words]
        return token_ids


class NewsAggCollator:
    def __init__(self, tokenizer: Tokenier, device="cuda"):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        max_length = max([item["length"] for item in batch])
        input_ids = []
        labels = []
        for item in batch:
            tokenized = self.tokenizer(item["title"])
            pad = [0] * (max_length - len(tokenized))
            input_ids.append(tokenized + pad)
            labels.append(item["category_id"])

        tensors = {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
        }
        return tensors

    def debug(self, dataset, batch_size=3):
        loader = DataLoader(
            dataset, collate_fn=self, batch_size=batch_size, shuffle=True
        )
        for batch in loader:
            print(batch)
            break


class NewsAggDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        # News category:
        #     b = business,
        #     t = science and technology,
        #     e = entertainment,
        #     m = health
        df["category_id"] = pd.Categorical(
            df["category"], categories=["b", "t", "e", "m"]
        ).codes
        df["length"] = df["title"].str.len()
        self.data = df.to_dict(orient="records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    tokenizer = Tokenier(vocab={})
    df = pd.read_table("./test.tsv")
    ds = NewsAggDataset(df)
    collator = NewsAggCollator(tokenizer)
    collator.debug(ds)
