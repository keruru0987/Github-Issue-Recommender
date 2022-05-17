# coding=utf-8
# @Author : Eric
import heapq

import settings
import pandas as pd


def calculate_Pk(rel_list, k):
    # 前k个推荐中的正确率
    rel_num = 0
    for i in range(0, k):
        if rel_list[i] > 0:
            rel_num += 1
    Pk = rel_num/k
    return Pk


def calculate_AP(rel_list, api):
    # 计算Average Precision
    N = settings.select_num  # 推荐的个数
    m = 0  # 总的相关个数
    fpath = settings.tagged_github_filepath[api]
    gi_df = pd.read_csv(fpath)
    gi_df = gi_df.fillna('')
    for index, row in gi_df.iterrows():
        if row['rel'] > 0:
            m += 1
    # print(m)
    score = 0
    for k in range(1, N+1):
        score += calculate_Pk(rel_list, k) * rel_list[k-1]
    if m < N:
        min_mn = m
    else:
        min_mn = N
    score = score/min_mn
    return score

# 按列表a中元素的值进行排序，并返回元素对应索引序列
a = [1, 3, 5, 5, 2, 7, 9]
print('a:', a)
sorted_id = sorted(range(len(a)), key=lambda k: a[k], reverse=True)
print('元素索引序列：', sorted_id)



test_list = [3, 2, 4, 4, 1]
a = heapq.nlargest(5, test_list)
b = map(test_list.index, a)
sort_re = list(b)
print(calculate_AP(test_list, 'TextBlob'))
