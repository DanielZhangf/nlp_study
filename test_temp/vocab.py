import time
import numpy as np
import random
import pandas as pd
from collections import Counter

def preprocess(text, freq=20):


    words = text.split()

    # 删除低频词，减少噪音影响
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > freq]

    return trimmed_words

with open('Test.txt',encoding='utf-8') as f:
    text1 = f.read()

with open('Train.txt',encoding='utf-8') as f:
    text2 = f.read()

with open('Train_report.txt',encoding='utf-8') as f:
    text3 = f.read()

print(len(text2))
print(len(text3))
print(len(text1))
text = text1+text2+text3
words = preprocess(text)
vocab = set(words)
len(vocab)
vocab_to_int = {word: index for index, word in enumerate(vocab)}
print(vocab_to_int)
print("total words: {}".format(len(words)))
print("unique words: {}".format(len(set(words))))
# int_words = [vocab_to_int[w] for w in words]

with open("vocab.txt",'w',encoding = 'utf-8') as f:
    for k,v in vocab_to_int.items():
        f.write("%s\t%d\n"%(k,v))

f.close()