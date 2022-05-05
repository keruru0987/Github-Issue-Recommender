# coding=utf-8
# @Author : Eric
import pandas

filepath = 'data/issue/allennlp.json'
df = pandas.read_json(filepath)
print(df.head())
