import torch

from lstm.src import config
from lstm.src.model import ReviewAnalyzeModel
from lstm.src.tokenizer import JiebaTokenizer


def predict_batch(input_tensor, model):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)
    return probs.tolist()
def predict(user_input, model, tokenizer, device):
    input_ids = tokenizer.encode(user_input, config.SEQ_LEN)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    probs = predict_batch(input_tensor, model)
    prob = probs[0]
    return prob
def run_predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')
    model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size,padding_idx=tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))
    print('请输入要预测的评论：（输入 q 或 quit 退出）')
    while True:
        user_input = input('> ')
        if user_input in ['q', 'quit']:
            print('退出程序')
            break
        if not user_input:
            print('输入为空，请重新输入')
            continue
        prob = predict(user_input, model, tokenizer, device)
        if prob > 0.5:
            print(f'正面评价（置信度：{prob:.2f}）')
        else:
            print(f'负面评价（置信度：{1 - prob:.2f}）')
