# coding=utf-8
# @Author : Eric
import matplotlib.pyplot as plt

x = [5, 10, 15, 20, 25, 30]
y = [0.725, 0.780, 0.702, 0.763, 0.606, 0.576]

plt.plot(x, y, marker='^', markersize=5)
plt.title('TextBlob my_AP@N')
plt.xlabel('N')
plt.ylabel('my_AP@N')
plt.show()