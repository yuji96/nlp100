import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput


class RNNUnit(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.linear_input = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_hidden = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        return self.relu(self.linear_input(x) + self.linear_hidden(h))


class RNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.word_emb = nn.Embedding(vocab_size, hidden_size)

        self.rnn = RNNUnit(hidden_size)
        self.linear_output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels, hidden_init=None):
        assert input_ids.ndim == 2
        batch_size, T = input_ids.size()
        if hidden_init is None:
            hidden_init = torch.zeros(
                (batch_size, self.hidden_size),
                device="cuda" if input_ids.is_cuda else "cpu",
            )
        assert hidden_init.ndim == 2

        h_t = hidden_init
        for t in range(T):
            x_t = self.word_emb(input_ids[:, t])
            h_t = self.rnn(x_t, h_t)
        logits = self.softmax(self.linear_output(h_t))
        loss = self.loss(logits, labels)
        return ModelOutput(loss=loss, logits=logits)
