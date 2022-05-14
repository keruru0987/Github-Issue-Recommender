# coding=utf-8
# @Author : Eric

from textblob import TextBlob

text = "My major is Criminal Justice Education"

print(TextBlob(text).sentiment)
