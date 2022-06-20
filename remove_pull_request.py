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
        columns=['html_url', 'number', 'labels', 'state', 'created_at', 'pull_request', 'comments', 'title', 'body'])

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
            temp_dict['comments'] = row['comments']
            new_df = new_df.append(temp_dict, ignore_index=True)

    new_df.to_csv('data/new_issue/' + nlp_api + '.csv', index=True)
    print(nlp_api + ' finish')


def combine_gidata():
    new_df = pd.DataFrame(
        columns=['html_url', 'number', 'labels', 'state', 'created_at', 'pull_request', 'comments', 'title', 'body'])
    for i in range(0, 7):
        nlp_api = settings.nlp_api[i]
        fpath = settings.new_github_filepath[nlp_api]
        old_df = pd.read_csv(fpath)
        temp_dict = {}
        for index, row in old_df.iterrows():
            temp_dict['html_url'] = row['html_url']
            temp_dict['number'] = row['number']
            temp_dict['title'] = row['title']
            temp_dict['labels'] = row['labels']
            temp_dict['state'] = row['state']
            temp_dict['created_at'] = row['created_at']
            temp_dict['body'] = row['body']
            temp_dict['pull_request'] = row['pull_request']
            temp_dict['comments'] = row['comments']
            new_df = new_df.append(temp_dict, ignore_index=True)
        print(nlp_api + ' finish')

    new_df.to_csv('data/new_issue/allissue.csv', index=True)


def combine_sodata():
    new_df = pd.DataFrame(
        columns=['Id', 'AcceptedAnswerId', 'Tags', 'Title', 'Body', 'Score', 'ViewCount', 'AnswerCount', 'CommentCount'])
    for i in range(0, 7):
        nlp_api = settings.nlp_api[i]
        fpath = settings.stackoverflow_filepath[nlp_api]
        old_df = pd.read_csv(fpath)
        temp_dict = {}
        for index, row in old_df.iterrows():
            temp_dict['Id'] = row['Id']
            temp_dict['AcceptedAnswerId'] = row['AcceptedAnswerId']
            temp_dict['Tags'] = row['Tags']
            temp_dict['Title'] = row['Title']
            temp_dict['Body'] = row['Body']
            temp_dict['Score'] = row['Score']
            temp_dict['ViewCount'] = row['ViewCount']
            temp_dict['AnswerCount'] = row['AnswerCount']
            temp_dict['CommentCount'] = row['CommentCount']
            new_df = new_df.append(temp_dict, ignore_index=True)
        print(nlp_api + ' finish')

    new_df.to_csv('data/stackoverflow/allso.csv', index=True)


if __name__ == '__main__':
    #for i in range(0, 7):
        #remove_pull_and_create(i)
    combine_sodata()




