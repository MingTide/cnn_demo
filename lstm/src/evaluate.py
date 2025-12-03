import torch

from lstm.src import config
from lstm.src.dataset import get_dataloader
from lstm.src.model import ReviewAnalyzeModel
from lstm.src.predict import predict_batch
from lstm.src.tokenizer import JiebaTokenizer


def evaluate(model, dataloader, device):
    total_count = 0
    correct_count = 0
    model.eval()
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.tolist()
        probs = predict_batch(inputs, model)
        for prob, target in zip(probs, targets):
            pred_label = 1 if prob > 0.5 else 0
            if pred_label == target:
                correct_count += 1
            total_count += 1
    return correct_count / total_count

def run_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')
    model = ReviewAnalyzeModel( vocab_size=tokenizer.vocab_size,padding_idx=tokenizer.pad_token_index ).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))
    dataloader = get_dataloader(train=False)
    acc = evaluate(model, dataloader, device)
    print("========== 评估结果 ==========")
    print(f"准确率：{acc:.4f}")
    print("=============================")

if __name__ == '__main__':
    run_evaluate()