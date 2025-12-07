
import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

import config
from tokenizer import ChineseTokenizer, EnglishTokenizer
from dataset import get_dataloader
from predict import predict_batch
from transformer.src.model import TranslationModel


def evaluate(dataloader, model, zh_tokenizer, en_tokenizer, device):
  all_references = []
  all_predictions = []
  special_tokens = [zh_tokenizer.pad_token_index, zh_tokenizer.eos_token_index, zh_tokenizer.sos_token_index]
  for src, tgt in tqdm(dataloader, desc="评估"):
      src = src.to(device)
      tgt = tgt.tolist()
      predict_indexes = predict_batch(src, model, en_tokenizer, device)
      all_predictions.extend(predict_indexes)
      for indexes in tgt:
        indexes = [index for index in indexes if index not in special_tokens]
        all_references.append([indexes])
  bleu = corpus_bleu(all_references, all_predictions)
  return bleu

def run_evaluate():
  """
  启动评估流程。
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'zh_vocab.txt')
  en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'en_vocab.txt')

  model = TranslationModel(zh_vocab_size=zh_tokenizer.vocab_size,
         en_vocab_size = en_tokenizer.vocab_size,
         zh_padding_index = zh_tokenizer.pad_token_index,
         en_padding_index = en_tokenizer.pad_token_index).to(device)
  model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))

  dataloader = get_dataloader(train=False)

  bleu = evaluate(dataloader, model, zh_tokenizer, en_tokenizer, device)

  print('========== 评估结果 ==========')
  print(f'BLEU: {bleu:.2f}')
  print('=============================')

if __name__ == '__main__':
  run_evaluate()