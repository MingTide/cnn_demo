import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import config

class InputMethodDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_json(file_path, lines=True).to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]['input'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['target'], dtype=torch.long)
        return input_tensor, target_tensor

def get_dataloader(train=True):
    file_name = 'train.jsonl' if train else 'test.jsonl'
    dataset = InputMethodDataset(config.PROCESSED_DATA_DIR / file_name)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

if __name__ == '__main__':
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(train=False)

