import  pandas as pd
from sklearn.model_selection import train_test_split



from lstm.src import config
from lstm.src.config import RAW_DATA_DIR
from lstm.src.tokenizer import JiebaTokenizer


def process():
    df = pd.read_csv(RAW_DATA_DIR /  'online_shopping_10_cats.csv',usecols=['review', 'label'],encoding='utf-8')
    df = df.dropna()
    df = df[df['review'].str.strip().ne('')]
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    JiebaTokenizer.build_vocab(train_df['review'].tolist(), config.PROCESSED_DATA_DIR / 'vocab.txt')
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')
    train_df['review'] = train_df['review'].apply(lambda x: tokenizer.encode(x, seq_len=config.SEQ_LEN))
    train_df.to_json(config.PROCESSED_DATA_DIR / 'indexed_train.jsonl',orient='records',lines=True)
    test_df['review'] = test_df['review'].apply(lambda x: tokenizer.encode(x, seq_len=config.SEQ_LEN))
    test_df.to_json(config.PROCESSED_DATA_DIR / 'indexed_test.jsonl',orient='records',lines=True)
    print("数据处理完成")

if __name__ == '__main__':
    process()
