import pandas as pd
from sklearn.pipeline import Pipeline #管道机制
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split #分训练和测试集
#导入“流水线”各个模块（标准化，降维，分类）
from  sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# 准备数据
data = pd.read_csv(r'Dataset.csv')
df = pd.DataFrame(data)
target = 'target'
features = df.columns.drop(target)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)


# # # 流水线
# # pipe=Pipeline(steps=[('standardScaler',StandardScaler()), ('pca', PCA()), ('svc',SVC())])
# pipe=make_pipeline(StandardScaler(),PCA(),SVC()) # 我们通常不需要为每一个步骤提供用户指定的名称，这种情况下，就可以用make_pipeline函数创建管道，它可以为我们创建管道并根据每个步骤所属的类为其自动命名。
# print(pipe.steps)
# pipe.fit(X_train, y_train) #训练模型
# pipe.predict(X_test) #预测结果
# print('Test accuracy: %.3f' % pipe.score(X_test, y_test))#输出精度



# 定义流水线+网格搜索参数
pipeline=Pipeline([('scaler',StandardScaler()),('pca',PCA()),('svm',SVC())])
param_grid={'svm__C':[0.001,0.01,0.1,1,10,100],'svm__gamma':[0.001,0.01,0.1,1,10,100]}# 定义网格搜索参数，用<estimator>__<parameter>形式设置参数
grid=GridSearchCV(pipeline,param_grid,cv=5, scoring='accuracy')# 网格搜索模型实例化
grid.fit(X_train,y_train)
grid.predict(X_test)
print('Test accuracy: %.3f' % grid.score(X_test, y_test))#输出精度
