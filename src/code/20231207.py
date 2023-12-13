import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


fea1 = np.array([1, 2, 3, 4, 5])
fea2 = np.array([3, 2, 1, 4, 3])
fea3 = np.array([5, 2, 3, 4, 1])
label = np.array([0, 1, 1, 0, 1])

# fea1 = [1, 2, 3, 4, 5]
# fea2 = [3, 2, 1, 4, 3]
# fea3 = [5, 2, 3, 4, 1]
# label = [0, 1, 1, 0, 1]

data_x = np.column_stack((fea1, fea2, fea3))
print(data_x)
data_y = label.reshape((len(label), 1))
print(data_y)


x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=1/5, random_state=0)
model = LinearRegression()

model.fit(x_train, y_train)
print(x_test)
y_pred_test = model.predict(x_test)
print(y_pred_test)