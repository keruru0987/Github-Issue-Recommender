# coding=utf-8
# @Author : Eric
import settings
import pandas as pd


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

        # selected_df = pd.DataFrame(columns=['ID', 'Title', 'Body'])
        result_so = []

        for index, row in sodata_df.iterrows():
            match_score = 0
            # 把所有字符转换为小写形式
            for word in self.query.lower().split():
                if word in row['Title'].lower():
                    match_score += 1
                if word in row['Tags'].lower():
                    match_score += 3
                # 这里还可以考虑body的影响

            if match_score >= 1:
                link = 'https://stackoverflow.com/questions/'+str(row['Id'])
                # 构造其中一条数据
                information = [link, row['Title'], row['Body'], match_score]
                result_so.append(information)

        return_so = sorted(result_so, reverse=True, key=lambda post: post[3])

        return return_so


if __name__ == '__main__':
    searcher = SOSearcher('TextBlob', 'install dataframe')
    so = searcher.search()
    for ele in so:
        print(ele)

