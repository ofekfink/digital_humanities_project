import torch
from torch import nn
from configuration import *


class Model(nn.Module):

    def __init__(self, vocab_size):
        super(Model, self).__init__()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True)
        self.linear = nn.Linear(
            in_features=num_directions * hidden_size,
            out_features=vocab_size)

    def forward(self, input):
        h = torch.zeros(num_layers * num_directions, batch_size, hidden_size, dtype=torch.double)
        c = torch.zeros(num_layers * num_directions, batch_size, hidden_size, dtype=torch.double)
        embed = self.embed(input.long())
        h, c = self.lstm(embed, (h, c))
        output = self.linear(h)
        output = output.permute(0, 2, 1)
        return output

