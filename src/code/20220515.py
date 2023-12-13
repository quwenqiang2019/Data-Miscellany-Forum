import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import xlrd
import seaborn as sns


def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)
    optimal_threshold = threshold[Youden_index]
    print(optimal_threshold)
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def acu_curve(index_name, y, prob):
    font = {'family': 'Times New Roman',
            'size': 12,
            }
    sns.set(font_scale=1.2)
    plt.rc('font', family='Times New Roman')
    fpr, tpr, thresholds = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    print(optimal_point)
    plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
    plt.text(optimal_point[0], optimal_point[1], (float('%.2f' % optimal_point[0]),
                                                  float('%.2f' % optimal_point[1])),
             ha='right', va='top', fontsize=12)
    plt.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th:.2f}', fontsize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC analysis of ' + index_name, fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.show()


def extract(inpath, sheet_index, label_index, col_index):
    data = xlrd.open_workbook(inpath, encoding_override='utf-8')
    table = data.sheets()[sheet_index]  # 选定表
    nrows = table.nrows  # 获取行号
    ncols = table.ncols  # 获取列号
    label = []
    prb = []
    for i in range(1, nrows):  # 第0行为表头
        alldata = table.row_values(i)  # 循环输出excel表中每一行，即所有数据
        label_value = alldata[label_index]
        label.append(label_value)
        result = alldata[col_index]  # 取出表中第二列数据
        prb.append(result)
    return label, prb


if __name__ == "__main__":
    inpath = 'F:\医学大数据课题\RA预测\RA预测\指标分析.xls'

    (label, prb) = extract(inpath, 0, 0, 1)
    print(label)
    print(prb)
    acu_curve(index_name='CCP', y=label, prob=prb)

    (label, prb) = extract(inpath, 1, 0, 1)
    print(label)
    print(prb)
    acu_curve(index_name='MCV', y=label, prob=prb)

    (label, prb) = extract(inpath, 2, 0, 1)
    print(label)
    print(prb)
    acu_curve(index_name='RF', y=label, prob=prb)

    (label, prb) = extract(inpath, 3, 0, 1)
    print(label)
    print(prb)
    acu_curve(index_name='CRP', y=label, prob=prb)
