# coding=utf-8
# @Author : Eric

from flask import Flask, render_template, request
from stackoverflow_search import SOSearcher, SOFinder
from githubissue_search import GIRecommend

app = Flask(__name__)


@app.route('/mainpage', methods=["GET", "POST"])
def mainpage():
    if request.method == 'GET':
        return render_template("github issue recommender.html")
    else:
        selected_api = request.form.get("selected_api")
        so_query = request.form.get("so_query")
        print(selected_api, so_query)

        # 需要构造一个列表，其中的每一项内容需要包括:链接（由ID号进行构造）,标题，内容, Id
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
    print(api)
    print(id)
    SO_Finder = SOFinder(api, id)
    matched_so = SO_Finder.find()  # title,body,tags

    GI_Recommender = GIRecommend(api, matched_so[0], matched_so[1], matched_so[2])
    result = GI_Recommender.recommend()
    # for inf in result:
    #     print(inf)

    return render_template('gi_results.html', u=result)


if __name__ == '__main__':
    app.run()
