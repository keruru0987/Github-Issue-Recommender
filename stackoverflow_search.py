# coding=utf-8
# @Author : Eric
import re

import settings
import pandas as pd
import data_process


class SOSearcher(object):
    def __init__(self, selected_api, so_query):
        self.api = selected_api
        print('当前api为：' + self.api)
        self.query = so_query

    def search(self):
        """
        根据给定的API和查询语句，返回与其相关的so posts，根据相关程度进行降序排序
        :return: 包含posts信息的列表，信息分别为链接（用ID进行构造），标题，内容，相关度
        """
        fpath = settings.stackoverflow_filepath[self.api]
        sodata_df = pd.read_csv(fpath)
        sodata_df = sodata_df.fillna('')

        # selected_df = pd.DataFrame(columns=['ID', 'Title', 'Body'])
        result_so = []

        for index, row in sodata_df.iterrows():
            match_score = 0
            # 把所有字符转换为小写形式
            for word in self.query.lower().split():
                if word in row['Title'].lower():
                    match_score += 1
                if word in row['Tags'].lower():
                    match_score += 1
                # 这里还可以考虑body的影响

            if match_score >= 1:
                link = 'https://stackoverflow.com/questions/'+str(row['Id'])
                # 构造其中一条数据
                acc_id = ''
                if row['AcceptedAnswerId'] != '':
                    acc_id = int(row['AcceptedAnswerId'])
                processed_body_text = data_process.process_query(row['Body'])  # 去掉code，图片的
                imgsize_fixed_body = data_process.alter_pic_size(row['Body'])  # 调整图片大小后的
                tag_str = row['Tags']
                tag_pattern = r'<.*?>'
                tags = re.findall(tag_pattern, tag_str)
                re_tag = ''
                for tag in tags:
                    tag1 = tag.replace('<', '')
                    tag2 = tag1.replace('>', '')
                    re_tag = re_tag + tag2 + ' '

                information = [link, row['Title'], imgsize_fixed_body, str(row['Id']), str(acc_id), processed_body_text,
                               str(row['Score']), str(row['ViewCount']), re_tag, str(row['AnswerCount']),
                               str(row['CommentCount']), match_score, row['ViewCount']]
                result_so.append(information)

        # 先根据热度进行排序
        so_hot_sorted = sorted(result_so, reverse=True, key=lambda post: post[12])
        # 再按照query相关度进行排序
        return_so = sorted(so_hot_sorted, reverse=True, key=lambda post: post[11])

        return return_so


class SOFinder(object):
    def __init__(self, selected_api, id_str):
        self.api = selected_api
        self.id_str = id_str

    def find(self):
        fpath = settings.stackoverflow_filepath[self.api]
        sodata_df = pd.read_csv(fpath)
        sodata_df = sodata_df.fillna('')
        matched_so = []
        match_flag = 0
        for index, row in sodata_df.iterrows():
            if str(row['Id']) == self.id_str:
                acc_id = ''
                if row['AcceptedAnswerId'] != '':
                    acc_id = int(row['AcceptedAnswerId'])
                matched_so = [row['Title'], row['Body'], row['Tags'], str(acc_id)]
                match_flag = 1
                break
        if match_flag == 0:
            raise Exception('没有匹配的ID')
        return matched_so


if __name__ == '__main__':
    searcher = SOSearcher('TextBlob', 'install error')
    so = searcher.search()
    for ele in so:
        print(ele)

    finder = SOFinder('allennlp', '68862752')
    print(finder.find())

