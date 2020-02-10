#coding:utf-8

from albert.extract_feature import BertVector
pooling_strategy = "REDUCE_MEAN"
# pooling_strategy = "NONE"
bc = BertVector(pooling_strategy=pooling_strategy, max_seq_len=80)
s1 = '谢谢你，每一个平凡的中国人'
v = bc.encode([s1])
v1 = v["encodes"][0]
print(v1)

