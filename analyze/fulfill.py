# coding=utf-8
# @Author : Eric
import pandas as pd

fpath = 'data/new_tagged_data/new_tagged_transformers.csv'

df = pd.read_csv(fpath)
df = df.fillna(0)

df.to_csv('data/new_tagged_data/new_transformers.csv', index=True)