import time

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.dataset import get_dataloader
from src.model import InputMethodModel
import config


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    total_loss = 0
    model.train()
    for inputs, targets in tqdm(dataloader, desc='训练'):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss
def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(False)
    vocab_size = 0
    with open( config.MODELS_DIR  / "vocab.txt", "r", encoding="utf-8") as f:
        vocab_size = [line.strip() for line in f.readlines()]
    model = InputMethodModel(len(vocab_size)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    model.to(device)
    writer = SummaryWriter(log_dir=config.LOG_DIR/time.strftime("%Y%m%d-%H%M%S"))
    best_loss = float('inf')
    for epoch in range(1,1+config.EPOCHS):
        print("Epoch {}/{}".format(epoch,config.EPOCHS))
        loss =  train_one_epoch(model,dataloader, loss_fn, optimizer,device)
        print(f"loss: {loss}")
        writer.add_scalar('loss',loss,epoch)
        # 记录训练结果
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'model.pt')
            print('模型保存成功！')
    writer.close()
if __name__ == '__main__':
    train()

