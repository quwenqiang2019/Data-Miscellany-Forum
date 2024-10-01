import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report

# y_true为真实值，y_pred为预测值（此处y_true和y_pred仅作举例，随便取的值，有0~4共5个类别。）
y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
          3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
y_pred = [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 3, 3, 3, 0, 2, 2, 3, 3, 3, 3,
          3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4]

# 1.计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print(cm)
conf_matrix = pd.DataFrame(cm, index=['1', '2', '3', '4', '5'], columns=['1', '2', '3', '4', '5'])  # 数据有5个类别
print(conf_matrix)
# 画出混淆矩阵
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
plt.figure()
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 14}, cmap="Blues")
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('confusion.pdf', bbox_inches='tight')
plt.show()

# 2.计算accuracy
print('accuracy_score', accuracy_score(y_true, y_pred))

# 3.计算多分类的precision、recall、f1-score分数
print('Micro precision', precision_score(y_true, y_pred, average='micro'))
print('Micro recall', recall_score(y_true, y_pred, average='micro'))
print('Micro f1-score', f1_score(y_true, y_pred, average='micro'))

print('Macro precision', precision_score(y_true, y_pred, average='macro'))
print('Macro recall', recall_score(y_true, y_pred, average='macro'))
print('Macro f1-score', f1_score(y_true, y_pred, average='macro'))

# 下面这个可以显示出每个类别的precision、recall、f1-score。
print('classification_report\n', classification_report(y_true, y_pred))