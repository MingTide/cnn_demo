import torch
from torch import nn
import config
from torchinfo import summary
class ReviewAnalyzeModel(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding( num_embeddings = vocab_size,embedding_dim=config.EMBEDDING_DIM,padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=config.EMBEDDING_DIM,hidden_size=config.HIDDEN_DIM, batch_first=True)
        self.linear = nn.Linear(in_features=config.HIDDEN_DIM, out_features=1)

    def forward(self, x):
        embed = self.embedding(x)
        output, _ = self.lstm(embed)
        result = self.linear(output[:, -1, :]).squeeze(dim=1)
        return result
if __name__ == '__main__':
    model = ReviewAnalyzeModel(vocab_size=1000, padding_idx=0)
    dummy_input = torch.randint( low=0, high=1000,size=(config.BATCH_SIZE, config.SEQ_LEN), dtype=torch.long)
    summary(model, input_data=dummy_input)