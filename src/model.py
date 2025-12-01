import torch
from torch import nn


import config
class InputMethodModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM)
        self.rnn = nn.RNN(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_DIM,
            batch_first=True,

        )
        self.liner = nn.Linear(config.HIDDEN_DIM, vocab_size)
    def forward(self,x):
        embed = self.embedding(x)
        output, _ = self.rnn(embed)
        result = self.liner(output[:, -1, :])
        return result
