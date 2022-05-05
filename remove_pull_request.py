# coding=utf-8
# @Author : Eric
# 去掉原始gi数据中包含的pull request，并且删掉没用的信息, 去掉body为空的那些

import pandas as pd
import settings


def remove_pull_and_create(select):
    nlp_api = settings.nlp_api[select]
    fpath = settings.github_filepath[nlp_api]

    old_df = pd.read_json(fpath)
    old_df = old_df.fillna('')

    temp_dict = {}
    new_df = pd.DataFrame(
        columns=['html_url', 'number', 'labels', 'state', 'created_at', 'pull_request', 'title', 'body'])

    # 去掉其中的pull,保留issue
    for index, row in old_df.iterrows():
        if row['pull_request'] == '' and row['body'] != '':
            temp_dict['html_url'] = row['html_url']
            temp_dict['number'] = row['number']
            temp_dict['title'] = row['title']
            temp_dict['labels'] = row['labels']
            temp_dict['state'] = row['state']
            temp_dict['created_at'] = row['created_at']
            temp_dict['body'] = row['body']
            temp_dict['pull_request'] = row['pull_request']
            new_df = new_df.append(temp_dict, ignore_index=True)

    new_df.to_csv('data/new_issue/' + nlp_api + '.csv')
    print(nlp_api + ' finish')


if __name__ == '__main__':
    for i in range(0, 7):
        remove_pull_and_create(i)




