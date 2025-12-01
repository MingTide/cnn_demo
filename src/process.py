import jieba
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import RAW_DATA_DIR
import  config

def build_dataset(sentences,word2index):
    index_sentence = [[word2index.get(token, 0) for token in jieba.lcut(sentence)] for sentence in
                            sentences]
    dataset = []
    for sentence in tqdm(index_sentence,desc="构建数据集"):
        for i in range(len(sentence) - config.SEQ_LEN):
            inputs = sentence[i:i + config.SEQ_LEN]
            target = sentence[i + config.SEQ_LEN]
            dataset.append({'input': inputs, 'target': target})
    return dataset

def process():
    df = pd.read_json(RAW_DATA_DIR / 'synthesized_.jsonl', orient="records",lines=True).sample(frac=0.1)
    sentences = []
    for dialog in df['dialog']:
        for sentence in dialog:
            sentences.append(sentence.split("：")[1])


    train_sentences ,test_sentences = train_test_split(sentences,test_size=0.2,random_state=42)

    vocab_set = set()
    for sentence in tqdm(train_sentences,desc="构建词表"):
        vocab_set.update(jieba.lcut(sentence))
    vocab_list = ['<unk>'] + list(vocab_set)
    with open(config.MODELS_DIR / 'vocab.txt', 'w',encoding='utf-8') as f:
        f.write('\n'.join(vocab_list))

    # 构建训练集

    word2index = {word: index for index,word in enumerate(vocab_list)}
    train_dataset = build_dataset(train_sentences,word2index)
    test_dataset = build_dataset(test_sentences,word2index)
    # 保存训练集
    pd.DataFrame(train_dataset).to_json(config.PROCESSED_DATA_DIR / 'train.jsonl', orient='records',lines=True)
    pd.DataFrame(test_dataset).to_json(config.PROCESSED_DATA_DIR / 'test.jsonl', orient='records',lines=True)

    pass
if __name__ == '__main__':
    process()