# coding=utf-8
# @Author : Eric

from flask import Flask, render_template, request

import data_process
import settings
from stackoverflow_search import SOSearcher, SOFinder
from githubissue_search import GIRecommend

app = Flask(__name__)


@app.route('/mainpage', methods=["GET", "POST"])
def mainpage():
    if request.method == 'GET':
        print('begin')
        return render_template("github issue recommender.html")
    else:
        selected_api = request.form.get("selected_api")
        so_query = request.form.get("so_query")
        print(selected_api, so_query)

        # 需要构造一个列表，其中的每一项内容需要包括:链接（由ID号进行构造）,标题，内容, Id, AcceptedAnswerId,
        # processed_body_text,score, viewcount, tag_list, AnswerCount, CommentCount
        SO_Seacher = SOSearcher(selected_api, so_query)
        result = SO_Seacher.search()

        return render_template("so_results.html", u=result, api=selected_api, query=so_query)


@app.route('/recommend')
def recommend():
    """
    根据从前端传入的StackOverflow Id号和api，为其推荐相关的Github Issue
    :return:
    """
    api = request.args.get("api")
    id = request.args.get("id")  # str形式的
    print("当前要进行推荐的api: " + api)
    print("当前要进行推荐的so_id: " + id)
    SO_Finder = SOFinder(api, id)
    matched_so = SO_Finder.find()
    # title,body,tags,AcceptedAnswerId, score, view_count, answer_count, comment_count, tag_list
    img_processed_body = data_process.alter_pic_size(matched_so[1])
    title = matched_so[0]
    acc_id = matched_so[3]
    score = matched_so[4]
    view_count = matched_so[5]
    answer_count = matched_so[6]
    comment_count = matched_so[7]
    tag_list = matched_so[8]

    GI_Recommender = GIRecommend(api, matched_so[0], matched_so[1], matched_so[2])
    result = GI_Recommender.recommend()  # link,title,body,number,state,clean_body,comments,tags
    # for inf in result:
    #     print(inf)

    label_pre_link = settings.api_label_prelink[api]

    return render_template('gi_results.html', u=result, img_processed_body=img_processed_body, title=title,
                           acc_id=acc_id, so_id=id, score=score, view_count=view_count, answer_count=answer_count,
                           comment_count=comment_count, tag_list=tag_list, tag_pre_link=label_pre_link)


if __name__ == '__main__':
    app.run()
