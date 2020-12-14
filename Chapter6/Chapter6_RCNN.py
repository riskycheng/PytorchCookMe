import torch
import random
from Commons import LocalUtils

with open('./jaychou_lyrics.txt', 'r', encoding='utf-8') as file:
    corpus_chars = file.read()
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[:10000]
print(corpus_chars[:40])

# 将每个字符映射成一个从0开始的连续整数
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
print(char_to_idx)
vocab_size = len(char_to_idx)
print(vocab_size)

# 将训练数据中每个字符转化为索引
corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:20]
print('char:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)

# 测试随机采样和相邻采样
for X, Y in LocalUtils.data_iter_consecutive(corpus_indices, batch_size=2, num_steps=6):
    print('X:', X, '\nY:', Y, '\n\n')

for X, Y in LocalUtils.data_iter_random(corpus_indices, batch_size=2, num_steps=6):
    print('X:', X, '\nY:', Y, '\n\n')
