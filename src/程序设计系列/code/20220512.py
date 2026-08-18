import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def Draw_ROC(file1,file2):
    data1=pd.read_excel(file1)
    data1=pd.DataFrame(data1)
    data2=pd.read_excel(file2)
    data2=pd.DataFrame(data2)

    fpr_CSNN,tpr_CSNN,thresholds=roc_curve(list(data1['真实标签']),
                                           list(data1['positive概率']))
    roc_auc_CSSSNN=auc(fpr_CSNN,tpr_CSNN)

    fpr_NN,tpr_NN,thresholds=roc_curve(list(data2['真实标签']),
                                       list(data2['positive概率']))
    roc_auc_DL=auc(fpr_NN,tpr_NN)

    font = {'family': 'Times New Roman',
            'size': 12,
            }
    sns.set(font_scale=1.2)
    plt.rc('font',family='Times New Roman')

    plt.plot(fpr_NN,tpr_NN,'purple',label='NN_AUC = %0.2f'% roc_auc_DL)
    plt.plot(fpr_CSNN,tpr_CSNN,'blue',label='CSNN_AUC = %0.2f'% roc_auc_CSSSNN)
    plt.legend(loc='lower right',fontsize = 12)
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('True Positive Rate',fontsize = 14)
    plt.xlabel('Flase Positive Rate',fontsize = 14)
    plt.show()

if __name__=="__main__":
    Draw_ROC('F:\医学大数据课题\RA预测\RA预测\CSSSDL_0.92-0.86-0.90.xlsx',
             'F:\医学大数据课题\RA预测\RA预测\DL_0.83-0.83-0.83.xlsx')
