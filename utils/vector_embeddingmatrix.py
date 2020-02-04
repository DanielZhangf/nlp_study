import re
import jieba
import pandas as pd
from utils.config import merger_seg_path,save_model_path,save_matrix_path,save_wordvector_path
import gensim
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec
import logging
import numpy as np


# 1.读取词汇库，并进行词向量训练
def train_model(path,save_model_path):
    merger_df = pd.read_csv(path,header=None)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    Word2VecModel = word2vec.Word2Vec(LineSentence(path), workers=8,min_count=5,size=200)
    print(Word2VecModel.wv.most_similar(['奇瑞'],topn=10))
    Word2VecModel.save(save_model_path)
    return Word2VecModel


#2.构造embedding_matrix
def embedding_matrix(Word2VecModel,save_matrix_path,save_wordvector_path):
    vocab_list = [word for word, Vocab in Word2VecModel.wv.vocab.items()]# 存储 所有的 词语
    print(type(vocab_list))
    print(len(vocab_list))
    word_index = {" ": 0} # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。使用前必须添加一个索引0.
    word_vector = {} # 初始化`[word : vector]`字典

    # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
    # 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如100。
    embedding_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))

    ## 3 填充 上述 的字典 和 大矩阵

    for i in range(len(vocab_list)):
        # print(i)
        word = vocab_list[i]  # 每个词语
        word_index[word] = i + 1 # 词语：索引
        word_vector[word] = Word2VecModel.wv[word] # 词语：词向量
        embedding_matrix[i + 1] = Word2VecModel.wv[word]  # 词向量矩阵

    print(embedding_matrix.shape)
    np.savetxt(save_matrix_path,embedding_matrix)
    np.save(save_wordvector_path, word_vector)

if __name__ == '__main__':
    # 数据集批量处理
    Word2VecModel = train_model(merger_seg_path,save_model_path)
    embedding_matrix(Word2VecModel,save_matrix_path,save_wordvector_path)

