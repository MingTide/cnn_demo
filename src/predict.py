import jieba
import torch

from src import config
from src.dataset import InputMethodDataset
from src.model import InputMethodModel


def predict(text):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(config.MODELS_DIR / 'vocab.txt', 'rb') as f:
        vocab_list = [line.strip() for line  in f.readlines()]
    word2index = {word:index for index,word in enumerate(vocab_list)}
    index2word = {index:word for index,word in enumerate(vocab_list)}
    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pth', map_location=device))
    tokens = jieba.lcut(text)
    indexes = [word2index.get(token, 0) for token in tokens]
    input_tensor = torch.tensor(indexes).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output=model(input_tensor)
        top5_indexed = torch.topk(output, k=5).indices
        top5_indexed_list = top5_indexed.tolist()
        top5 = [index2word[index] for index in top5_indexed_list[0]]
        return  top5


if __name__ == '__main__':
    l = predict("我们团队")
    print(l)



