import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import random
import seaborn as sns

def GWO(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    # ===初始化 alpha, beta, and delta_pos=======
    Alpha_pos = np.zeros(dim)  # 位置.形成30的列表
    Alpha_score = float("inf")  # 这个是表示“正负无穷”,所有数都比 +inf 小；正无穷：float("inf"); 负无穷：float("-inf")

    Beta_pos = np.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = np.zeros(dim)
    Delta_score = float("inf")  # float() 函数用于将整数和字符串转换成浮点数。

    # ====list列表类型=============
    if not isinstance(lb, list):  # 作用：来判断一个对象是否是一个已知的类型。其第一个参数（object）为对象，第二个参数（type）为类型名，若对象的类型与参数二的类型相同则返回True
        lb = [lb] * dim  # 生成[100，100，.....100]30个
    if not isinstance(ub, list):
        ub = [ub] * dim

    # ========初始化所有狼的位置===================
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):  # 形成5*30个数[-100，100)以内
        Positions[:, i] = np.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[
            i]  # 形成[5个0-1的数]*100-（-100）-100
    Convergence_curve = np.zeros(Max_iter)

    # ========迭代寻优=====================
    for l in range(0, Max_iter):  # 迭代1000
        for i in range(0, SearchAgents_no):  # 5
            # ====返回超出搜索空间边界的搜索代理====
            for j in range(dim):  # 30
                Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[
                    j])  # clip这个函数将将数组中的元素限制在a_min(-100), a_max(100)之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min。

            # ===计算每个搜索代理的目标函数==========
            fitness = objf(Positions[i, :])  # 把某行数据带入函数计算
            # print("经过计算得到：",fitness)

            # ====更新 Alpha, Beta, and Delta================
            if fitness < Alpha_score:
                Alpha_score = fitness  # Update alpha
                Alpha_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness < Beta_score):
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score):
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()

        # ===========以上的循环里，Alpha、Beta、Delta===========
        a = 2 - l * ((2) / Max_iter)  # a从2线性减少到0

        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()  # r1 is a random number in [0,1]主要生成一个0-1的随机浮点数。
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a  # Equation (3.3)
                C1 = 2 * r2  # Equation (3.4)
                # D_alpha表示候选狼与Alpha狼的距离
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[
                    i, j])  # abs() 函数返回数字的绝对值。Alpha_pos[j]表示Alpha位置，Positions[i,j])候选灰狼所在位置
                X1 = Alpha_pos[j] - A1 * D_alpha  # X1表示根据alpha得出的下一代灰狼位置向量

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a  #
                C2 = 2 * r2

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta

                Positions[i, j] = (X1 + X2 + X3) / 3  # 候选狼的位置更新为根据Alpha、Beta、Delta得出的下一代灰狼地址。

        Convergence_curve[l] = Alpha_score

        if (l % 1 == 0):
            print(['迭代次数为' + str(l) + ' 的迭代结果' + str(Alpha_score)])  # 每一次的迭代结果
    # 绘图
    plt.plot(Convergence_curve)
    plt.title('Convergence_curve')
    plt.show()

    print("The best solution obtained by GWO is : " + str(Alpha_pos))
    print("The best optimal value of the objective funciton found by GWO is : " + str(Alpha_score))
    return Alpha_pos, Alpha_score







if __name__ == "__main__":
    '''
    神经网络第一层神经元个数
    神经网络第二层神经元个数
    dropout比率
    batch_size
    '''
    # 读取数据集
    data = pd.read_csv('data.csv')
    # 将日期列转换为日期时间类型
    data['Month'] = pd.to_datetime(data['Month'])
    # 将日期列设置为索引
    data.set_index('Month', inplace=True)

    # 划分训练集和测试集
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # 绘制训练集和测试集的折线图
    sns.set(font_scale=1.2)
    plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
    plt.figure(figsize=(10, 6))
    plt.plot(train_data, label='Training Data')
    plt.plot(test_data, label='Testing Data')
    plt.xlabel('Year')
    plt.ylabel('Passenger Count')
    plt.title('International Airline Passengers - Training and Testing Data')
    plt.legend()
    plt.show()


    scaler = MinMaxScaler()
    train_data_scaler = scaler.fit_transform(train_data.values.reshape(-1, 1))
    test_data_scaler = scaler.transform(test_data.values.reshape(-1, 1))

    # 定义滑动窗口大小
    window_size = 1

    # 创建滑动窗口数据集
    X_train, y_train = create_sliding_windows(train_data_scaler, window_size)
    X_test, y_test = create_sliding_windows(test_data_scaler, window_size)

    # 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
    X_train = np.reshape(X_train, (X_train.shape[0], window_size, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], window_size, 1))


    ub = [51, 6, 0.055, 9]
    lb = [50, 5, 0.05, 8]

    # 开始优化===========主程序================
    Max_iter = 3  # 迭代次数
    dim = 4  # 狼的寻值范围
    SearchAgents_no = 5  # 寻值的狼的数量
    Alpha_pos, Alpha_score = GWO(training, lb, ub, dim, SearchAgents_no, Max_iter)

    print('best_params is ', Alpha_pos)
    print('best_precision is', Alpha_score)

    # 训练模型  使用GWO找到的最好的神经元个数
    neurons1 = int(Alpha_pos[0])
    neurons2 = int(Alpha_pos[1])
    dropout = Alpha_pos[2]
    batch_size = int(Alpha_pos[3])

    model = build_model(X_train, neurons1, neurons2, dropout)
    history1 = model.fit(X_train, y_train, epochs=150, batch_size=batch_size, validation_split=0.2, verbose=1,
                         callbacks=[EarlyStopping(monitor='val_loss', patience=9, restore_best_weights=True)])


    # 使用 LSTM 模型进行预测
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)


    # 反归一化预测结果
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)

    # 绘制测试集预测结果的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(test_data, label='Actual')
    plt.plot(list(test_data.index)[-len(test_predictions):], test_predictions, label='Predicted')
    plt.xlabel('Month')
    plt.ylabel('Passengers')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.show()

    # 绘制原始数据、训练集预测结果和测试集预测结果的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Actual')
    plt.plot(list(train_data.index)[window_size:train_size], train_predictions, label='Training Predictions')
    plt.plot(list(test_data.index)[-(len(test_data)-window_size):], test_predictions, label='Testing Predictions')
    plt.xlabel('Year')
    plt.ylabel('Passenger Count')
    plt.title('International Airline Passengers - Actual vs Predicted')
    plt.legend()
    plt.show()