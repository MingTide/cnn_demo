import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import RAW_DATA_DIR
import  config
from cnn.src.tokenizer import JiebaTokenizer


def build_dataset(sentences,jieba_tokenizer):
    index_sentence = [jieba_tokenizer.encode(sentence) for sentence in
                            sentences]
    dataset = []
    for sentence in tqdm(index_sentence,desc="构建数据集"):
        for i in range(len(sentence) - config.SEQ_LEN):
            inputs = sentence[i:i + config.SEQ_LEN]
            target = sentence[i + config.SEQ_LEN]
            dataset.append({'input': inputs, 'target': target})
    return dataset

def process():
    df = pd.read_json(RAW_DATA_DIR / 'synthesized_.jsonl', orient="records",lines=True)
    sentences = []
    for dialog in df['dialog']:
        for sentence in dialog:
            sentences.append(sentence.split("：")[1])
    JiebaTokenizer.build_vocab(sentences, config.MODELS_DIR / 'vocab.txt')
    jieba_tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    train_sentences ,test_sentences = train_test_split(sentences,test_size=0.2,random_state=42)
    # 构建训练集
    train_dataset = build_dataset(train_sentences,jieba_tokenizer)
    # 保存训练集
    pd.DataFrame(train_dataset).to_json(config.PROCESSED_DATA_DIR / 'train.jsonl', orient='records', lines=True)
    test_dataset = build_dataset(test_sentences,jieba_tokenizer)
    pd.DataFrame(test_dataset).to_json(config.PROCESSED_DATA_DIR / 'test.jsonl', orient='records', lines=True)
if __name__ == '__main__':
    process()