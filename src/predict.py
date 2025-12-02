import jieba
import torch

from src import config
from src.dataset import InputMethodDataset
from src.model import InputMethodModel
from src.tokenizer import JiebaTokenizer


def predict_batch(model,inputs):
    model.eval()
    with torch.no_grad():
        output = model(inputs)
        top5_indexed = torch.topk(output, k=5).indices
        top5_indexed_list = top5_indexed.tolist()
        return top5_indexed_list
def predict(text,model,tokenizer,device):


    indexes = tokenizer.encode(text)
    input_tensor = torch.tensor(indexes).unsqueeze(0).to(device)
    top5_indexed_list = predict_batch(model,input_tensor)
    top5 = [tokenizer.index2word[index] for index in top5_indexed_list[0]]
    return  top5

def run_predict():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = JiebaTokenizer.from_vocab( config.MODELS_DIR / 'vocab.txt')
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt', map_location=device))
    print("请输入（输入q或quit推出）")
    input_history = ''
    while True:
        user_input = input("> ")
        if user_input in ['q', 'quit']:
            break
        if user_input.strip() == '':
            print("请输入")
            continue
        input_history += user_input
        top5 = predict(input_history,model,tokenizer,device)
        print(top5)


if __name__ == '__main__':
    run_predict()



