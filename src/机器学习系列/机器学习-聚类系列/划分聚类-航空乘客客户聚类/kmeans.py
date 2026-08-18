import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns


# ==================数据读取===================
data = pd.read_csv('air_data.csv', sep=',', encoding='ANSI')
print(data)

# =================数据清洗============================
# 丢弃票价为0、平均折扣率不为0、总飞行公里数大于0的数据。
data = data[data['SUM_YR_1'].notnull() * data['SUM_YR_2'].notnull()]  # 票价非空值才保留
index1 = data['SUM_YR_1'] != 0
index2 = data['SUM_YR_2'] != 0
index3 = (data['SEG_KM_SUM'] == 0) & (data['avg_discount'] == 0)  # 该规则是“与”
data = data[index1 | index2 | index3]  # 该规则是“或”
print(data)

# =================属性规约============================
# 原始数据中属性太多，选择与其相关的六个属性，删除不相关、弱相关或冗余的属性。
data = data[["FFP_DATE", "LOAD_TIME", "FLIGHT_COUNT", "SUM_YR_1", "SUM_YR_2", "SEG_KM_SUM", "AVG_INTERVAL", "MAX_INTERVAL", "avg_discount"]]
print(data)



# ===================数据转换============================
# 属性构造、数据标准化。本模型将以下6个特征作为识别客户价值的特征。
data["LOAD_TIME"] = pd.to_datetime(data["LOAD_TIME"])
data["FFP_DATE"] = pd.to_datetime(data["FFP_DATE"])
data["入会时间"] = data["LOAD_TIME"] - data["FFP_DATE"]
data["平均每公里票价"] = (data["SUM_YR_1"] + data["SUM_YR_2"]) / data["SEG_KM_SUM"]
data["时间间隔差值"] = data["MAX_INTERVAL"] - data["AVG_INTERVAL"]
data = data.rename(columns = {"FLIGHT_COUNT" : "飞行次数", "SEG_KM_SUM" : "总里程", "avg_discount" : "平均折扣率"},inplace = False)
data = data[["入会时间", "飞行次数", "平均每公里票价", "总里程", "时间间隔差值", "平均折扣率"]]
data['入会时间'] = data['入会时间'].astype(np.int64)/(60*60*24*10**9)
data = (data - data.mean(axis=0))/(data.std(axis=0))
print(data)

# ===================构建模型============================
def distEclud(vecA, vecB):
    """
    计算两个向量的欧式距离的平方，并返回
    """
    return np.sum(np.power(vecA - vecB, 2))


def test_Kmeans_nclusters(data_train):
    """
    计算不同的k值时，SSE的大小变化
    """
    data_train = data_train.values
    nums = range(2, 10)
    SSE = []
    for num in nums:
        sse = 0
        kmodel = KMeans(n_clusters=num)
        kmodel.fit(data_train)
        # 簇中心
        cluster_ceter_list = kmodel.cluster_centers_
        # 个样本属于的簇序号列表
        cluster_list = kmodel.labels_.tolist()
        for index in range(len(data)):
            cluster_num = cluster_list[index]
            sse += distEclud(data_train[index, :], cluster_ceter_list[cluster_num])
        print("簇数是", num, "时； SSE是", sse)
        SSE.append(sse)
    return nums, SSE

# 采用计算SSE的方法，尝试找到最好的K数值
nums, SSE = test_Kmeans_nclusters(data)


#画图，通过观察SSE与k的取值尝试找出合适的k值
# 可视化部分
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
## 绘图观测SSE与簇个数的关系
fig=plt.figure(figsize=(10, 8))
ax=fig.add_subplot(1,1,1)
ax.plot(nums,SSE,marker="+")
ax.set_xlabel("n_clusters", fontsize=18)
ax.set_ylabel("SSE", fontsize=18)
fig.suptitle("KMeans", fontsize=20)
plt.show()

kmodel = KMeans(n_clusters=5)
kmodel.fit(data)
# 简单打印结果
r1 = pd.Series(kmodel.labels_).value_counts()  # 统计各个类别的数目
r2 = pd.DataFrame(kmodel.cluster_centers_)  # 找出聚类中心
# 所有簇中心坐标值中最大值和最小值
max = r2.values.max()
min = r2.values.min()
r = pd.concat([r2, r1], axis=1)  # 横向连接（0是纵向），得到聚类中心对应的类别下的数目
r.columns = list(data.columns) + [u'类别数目']  # 重命名表头

# 绘图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, polar=True)
center_num = r.values
feature = ["入会时间", "飞行次数", "平均每公里票价", "总里程", "时间间隔差值", "平均折扣率"]
N = len(feature)
feature = np.concatenate((feature, [feature[0]]))
for i, v in enumerate(center_num):
    # 设置雷达图的角度，用于平分切开一个圆面
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # 为了使雷达图一圈封闭起来，需要下面的步骤
    center = np.concatenate((v[:-1], [v[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    # 绘制折线图
    ax.plot(angles, center, 'o-', linewidth=2, label="第%d簇人群,%d人" % (i + 1, v[-1]))
    # 填充颜色
    ax.fill(angles, center, alpha=0.25)
    # 添加每个特征的标签
    ax.set_thetagrids(angles * 180 / np.pi, feature, fontsize=15)
    # 设置雷达图的范围
    ax.set_ylim(min - 0.1, max + 0.1)
    # 添加标题
    plt.title('客户群特征分析图', fontsize=20)
    # 添加网格线
    ax.grid(True)
    # 设置图例
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), ncol=1, fancybox=True, shadow=True)

# 显示图形
plt.show()



'''
通过观察可知：

当k取值4时，每个人群包含的信息比较复杂，且特征不明显
当k取值5时，分析的结果比较合理，分出的五种类型人群都有自己的特点又不相互重复
当k取值6时，各种人群也都有自己的特点，但是第4簇人群完全在第5簇人群特征中包含了，有点冗余的意思
综上，当k取值为5时，得到最好的聚类效果，将所有的客户分成5个人群，再进一步分析可以得到以下结论：

1.第一簇人群，10957人，最大的特点是时间间隔差值最大，分析可能是“季节型客户”，一年中在某个时间段需要多次乘坐飞机进行旅行，其他的时间则出行的不多，这类客户我们需要在保持的前提下，进行一定的发展；
2.第二簇人群，14732人，最大的特点就是入会的时间较长，属于老客户按理说平均折扣率应该较高才对，但是观察窗口的平均折扣率较低，而且总里程和总次数都不高，分析可能是流失的客户，需要在争取一下，尽量让他们“回心转意”；
3.第三簇人群，22188人，各方面的数据都是比较低的，属于一般或低价值用户
4.第三簇人群，8724人，最大的特点就是平均每公里票价和平均折扣率都是最高的，应该是属于乘坐高等舱的商务人员，应该重点保持的对象，也是需要重点发展的对象，另外应该积极采取相关的优惠政策是他们的乘坐次数增加
5.第五簇人群，5443人， 总里程和飞行次数都是最多的，而且平均每公里票价也较高，是重点保持对象
'''

