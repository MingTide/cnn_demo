import torch
from torch.utils.data import DataLoader


class DebugDataset(torch.utils.data.Dataset):
    def __init__(self, size=10):
        self.data = list(range(size))
        print(f"数据集初始化，有 {size} 个样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        print(f"加载第 {idx} 个样本: {item}")
        return item


print("创建DataLoader...")
debug_dataloader = DataLoader(DebugDataset(6), batch_size=2, shuffle=False)

print("开始迭代:")
for i, batch in enumerate(debug_dataloader):
    print(f"处理批次 {i}: {batch}")
    print("---")