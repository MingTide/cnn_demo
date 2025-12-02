import torch
from torch import nn


import config
class InputMethodModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        # print("Embedding权重形状:", embedding.weight.shape)  # [10000, 100]
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM)
        self.rnn = nn.RNN(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_DIM,
            batch_first=True,

        )
        self.liner = nn.Linear(config.HIDDEN_DIM, vocab_size)

        #x  [batch_size, seq_length]
    def forward(self,x):
        # embed 形状: [batch_size, seq_length, embedding_dim]
        embed = self.embedding(x)
        # output 形状: [batch_size, seq_length, hidden_size]
        output, _ = self.rnn(embed)
        # 3. 获取序列最后一个时间步的输出
        # output[:, -1, :] 形状: [batch_size, hidden_size]
        result = self.liner(output[:, -1, :])
        # result 形状: [batch_size, num_classes] 或 [batch_size, 1]
        return result
