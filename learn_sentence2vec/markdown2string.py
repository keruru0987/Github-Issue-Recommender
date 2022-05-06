# coding=utf-8
# @Author : Eric

from bs4 import BeautifulSoup
from markdown import markdown

t = '''

``` python
>>> from textblob import TextBlob
>>> b = TextBlob("I havv goood speling!")
>>> b.correct()
TextBlob("")
>>> print(b.correct())

>>> 
```'''

html = markdown(t)
text = BeautifulSoup(html, 'html.parser').get_text()
print(text)
