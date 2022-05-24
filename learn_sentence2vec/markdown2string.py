# coding=utf-8
# @Author : Eric

from bs4 import BeautifulSoup
from markdown import markdown

# t = '''
#
# ``` python
# >>> from textblob import TextBlob
# >>> b = TextBlob("I havv goood speling!")
# >>> b.correct()
# TextBlob("")
# >>> print(b.correct())
#
# >>>
# ```'''
#
# html = markdown(t)
# text = BeautifulSoup(html, 'html.parser').get_text()
# print(text)

import numpy as np
import matplotlib.pyplot as plt

def plotsub(x, y, num):
    plt.plot(x, y, label=num)


x = np.arange(0,10,1);
y = x*x

plotsub(x,y,'first')
plotsub(x*10,y,'second')

plt.show()
