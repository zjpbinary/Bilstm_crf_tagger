from pprint import pprint
from torchtext import data
from torchtext import datasets
from config import *
# Define the fields associated with the sequences.
WORD = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)
# TODO 字母级别信息对词性任务十分有用，可以提高许多潜在的信息，请同学们根据 https://github.com/pytorch/text/blob/master/test/sequence_tagging.py#L55 为模型加入字信息
PTB_TAG = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)

# Download and the load default data.
train, val, test = datasets.UDPOS.splits(
    fields=(('word', WORD), (None, None), ('ptbtag', PTB_TAG)))
pprint(train.fields)
pprint(len(train))
# 建立词表
WORD.build_vocab(train.word, min_freq=3)
PTB_TAG.build_vocab(train.ptbtag)

vocab = WORD.vocab
#print(train[0].word)
#print(train[0].ptbtag)
print()
unk_index = WORD.vocab.unk_index
pad_index = WORD.vocab.stoi['<pad>']

print(f"<UNK>: {unk_index}")
print(f"<pad>: {pad_index}")

n_words = len(WORD.vocab.itos)
n_tags = len(PTB_TAG.vocab.itos)
print(f"Word size: {n_words}")
print(f"Tag size: {n_tags}")

train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=args.batch_size, sort_within_batch=True)
#若需要使用pack_padded_sequence,sort_within_batch 设置为True

#print(PTB_TAG.vocab.stoi['<bos>'])
#print(PTB_TAG.vocab.stoi['<eos>'])

#batch = next(iter(train_iter))
# 默认返回的矩阵
#print(f"Data shape: {batch.word.shape}")
#print("words: \n", batch.word[:3])
#print("ptbtags: \n", batch.ptbtag[:3])

