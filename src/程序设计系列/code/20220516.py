import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def Draw_Parallel_bar(features,L1,L2,L3,L4,L5):
    features=features
    RA=L1;SS=L2;SLE=L3;Arthritis=L4;CTD=L5

    bar_width = 0.1
    index_RA = np.arange(len(features))
    index_SS = index_RA + bar_width
    index_SLE = index_SS + bar_width
    index_Arthritis = index_SLE + bar_width
    index_CTD = index_Arthritis + bar_width

    font = {'family': 'Times New Roman',
            'size': 12,
            }
    sns.set(font_scale=1.2)
    plt.rc('font',family='Times New Roman')

    plt.bar(index_RA, height=RA, width=bar_width, color='peru', label='RA')
    plt.bar(index_SS, height=SS, width=bar_width, color='g', label='SS')
    plt.bar(index_SLE, height=SLE, width=bar_width, color='y', label='SLE')
    plt.bar(index_Arthritis, height=Arthritis, width=bar_width, color='c', label='OA')
    plt.bar(index_CTD, height=CTD, width=bar_width, color='violet', label='CTDs')

    plt.legend(fontsize = 12)  # 显示图例
    plt.xticks(index_RA + 10*bar_width/5, features,fontsize = 12)
    plt.xlabel('Autoantibodies',fontsize = 14)
    plt.ylabel('Positive rate',fontsize = 14)
    plt.show()

if __name__=="__main__":
    Draw_Parallel_bar(('CCP', 'MCV', 'AKA', 'RF', 'ANA','CRP','ESR'),
                      [0.9265,0.8840,0.4816,0.8779,0.6167,0.6012,0.8562],
                      [0.1860,0.3256,0.1163,0.7674,0.7442,0.2558,0.5814],
                      [0.1905,0.3810,0.0476,0.6667,0.8571,0.3333,0.8571],
                      [0.4211,0.7368,0.2105,0.6316,0.5789,0.4737,0.7368],
                      [0.2308,0.3077,0.0769,0.8462,0.8468,0.3846,0.7692])
