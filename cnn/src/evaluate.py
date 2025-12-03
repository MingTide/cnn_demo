import torch
from tqdm import tqdm

from cnn.src import config
from cnn.src.dataset import get_dataloader
from cnn.src.model import InputMethodModel
from cnn.src.predict import predict_batch
from cnn.src.tokenizer import JiebaTokenizer


def evaluate(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        total_count = 0
        top1_correct = 0
        top5_correct = 0
        for inputs, targets in tqdm(dataloader, desc='评估'):
            #  [batch,sq_len] 64x5 64个batch 每个batch 5个字
            inputs = inputs.to(device)
            # [batch_size]
            targets = targets.tolist()
            # [batch_size,5]
            predicted_ids = predict_batch(model,inputs )
            for pred, target in zip(predicted_ids, targets):
                total_count += 1
                if pred[0] == target:
                    top1_correct += 1
                if target in pred:
                    top5_correct += 1

    top1_acc = top1_correct / total_count
    topk_acc = top5_correct / total_count
    return top1_acc, topk_acc
def run_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print("词表加载成功")
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt', map_location=device))
    test_dataloader = get_dataloader(False)
    # 评估逻辑
    top1_acc, top5_acc = evaluate(model, test_dataloader, device)
    print(top1_acc, top5_acc)

if __name__ == '__main__':
    run_evaluate()