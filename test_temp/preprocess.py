import jieba
from jieba import posseg
import time
import numpy as np

import random
import pandas
from collections import Counter
# def read_stopwords(path):
#     lines = set()
#     with open(path,mode = 'r',encoding='utf-8') as f:
#         for temp_line in f:
#             if isinstance(temp_line,str):
#             temp_line = temp_line.strip()
#             lines.add(temp_line)
#     return lines
#
train_file_path = "AutoMaster_TrainSet.csv"
test_file_path = "AutoMaster_TestSet.csv"
w_train_path = "Train.txt"
w2_train_path = "Train_report.txt"
w_test_path = "Test.txt"

def read_data(path):
    data = pandas.read_csv(path,encoding='utf-8')

    data_x = data["Question"].str.cat(data["Dialogue"])
    data_y = []
    if "Report" in data.columns:
        data_y = data["Report"]
    return data_x,data_y

x_train_data,y_train_data = read_data(train_file_path)
test_data,_ = read_data(test_file_path)

def write_data(path,text_data):
    with open(path,'w',encoding='utf-8') as f:
        for line_data in text_data:
            if isinstance(line_data, str):
                temp_line = line_data.replace("[语音]", "")
                temp_line = temp_line.replace("[图片]", "")
                temp_line = temp_line.replace("|", " ")
                temp_line = temp_line.replace("：", " ")
                temp_line = temp_line.replace("，", " ")
                temp_line = temp_line.replace("。", " ")
                temp_line = temp_line.replace("？", " ")
                temp_line = temp_line.replace("！", " ")
                seg_list = jieba.cut(temp_line)
                seg_line = ' '.join(seg_list)
                print(seg_line)
                f.write("%s"%seg_line)
            f.write("\n")


write_data(w_train_path,x_train_data)
write_data(w2_train_path,y_train_data)
write_data(w_test_path,test_data)
