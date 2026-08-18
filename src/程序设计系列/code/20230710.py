from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 假设你的字符串列表是 words_list
words_list = ["Python", "数据分析", "机器学习", "数据分析","Python","可视化", "NLP", "深度学习","Python", "数据分析", "数据挖掘", "Python","统计学"]

# 将列表中的字符串拼接成一个长字符串
words_str = " ".join(words_list)

# 创建词云对象
wordcloud = WordCloud(background_color="white", width=800, height=400, font_path='simsun.ttc').generate(words_str)

# 绘制词云图
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()