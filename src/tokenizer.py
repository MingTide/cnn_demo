import jieba
from tqdm import tqdm
jieba.setLogLevel(jieba.logging.WARNING)
class JiebaTokenizer:

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}
        self.unk_token_index = self.word2index[self.unk_token]
    unk_token = '<unk>'

    @staticmethod
    def tokenize(sentence):
        return jieba.cut(sentence)

    @classmethod
    def build_vocab(cls, sentences, vocab_file):
        unique_words = set()
        for sentence in tqdm(sentences, desc='分词'):
            for word in cls.tokenize(sentence):
                 unique_words.add(word)
        vocab_list = [cls.unk_token] + list(unique_words)
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for word in vocab_list:
                f.write(word + '\n')

    @classmethod
    def from_vocab(cls, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]
            return cls(vocab_list)

    def encode(self, sentence):
        tokens = self.tokenize(sentence)
        return [self.word2index.get(token, self.unk_token_index) for token in tokens]