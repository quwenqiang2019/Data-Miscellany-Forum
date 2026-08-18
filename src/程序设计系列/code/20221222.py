import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from collections import Counter
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import itertools
from pylab import mpl
import seaborn as sns

class Solution():
    #==================读取图片=================================
    def read_image(self,paths):
        os.listdir(paths)
        filelist = []
        for root, dirs, files in os.walk(paths):
            for file in files:
                if os.path.splitext(file)[1] == ".png":
                    filelist.append(os.path.join(root, file))
        return filelist

    #==================图片数据转化为数组==========================
    def im_array(self,paths):
        M=[]
        for filename in paths:
            im=Image.open(filename)
            im_L=im.convert("L")                #模式L
            Core=im_L.getdata()
            arr1=np.array(Core,dtype='float32')/255.0
            list_img=arr1.tolist()
            M.extend(list_img)
        return M

    def CNN_model(self,train_images, train_lables):
        # ============构建卷积神经网络并保存=========================
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))  # 过滤器个数，卷积核尺寸，激活函数，输入形状
        model.add(layers.MaxPooling2D((2, 2)))  # 池化层
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())  # 降维
        model.add(layers.Dense(64, activation='relu'))  # 全连接层
        model.add(layers.Dense(2, activation='softmax'))  # 注意这里参数，我只有两类图片，所以是2.
        model.summary()  # 显示模型的架构
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

if __name__=='__main__':
    Object1=Solution()
    # =================数据读取===============
    path1="D:\DCTDV2\dataset\\train\\"
    test1 = "D:\DCTDV2\dataset\\test\\"

    pathDir = os.listdir(path1)
    pathDir=pathDir[1:5]

    for a in pathDir:
        path2=path1+a
        test2=test1+a
        filelist_1=Object1.read_image(path1+"Norm")
        filelist_2=Object1.read_image(path2)
        filelist_all=filelist_1+filelist_2
        M=Object1.im_array(filelist_all)
        train_images=np.array(M).reshape(len(filelist_all),128,128)#输出验证一下(400, 128, 128)
        label=[0]*len(filelist_1)+[1]*len(filelist_2)
        train_lables=np.array(label)        #数据标签
        train_images = train_images[..., np.newaxis]        #数据图片
        print(train_images.shape)#输出验证一下(400, 128, 128, 1)

        # ===================准备测试数据==================
        filelist_1T = Object1.read_image(test1+"Norm")
        filelist_2T = Object1.read_image(test2)
        filelist_allT = filelist_1T + filelist_2T
        N = Object1.im_array(filelist_allT)
        dict_label = {0: 'norm', 1: 'IgaK'}
        test_images = np.array(N).reshape(len(filelist_allT), 128, 128)
        label = [0] * len(filelist_1T) + [1] * len(filelist_2T)
        test_lables = np.array(label)  # 数据标签
        test_images = test_images[..., np.newaxis]  # 数据图片
        print(test_images.shape)  # 输出验证一下(100, 128, 128, 1)

        # #===================训练模型=============
        model=Object1.CNN_model(train_images, train_lables)
        CnnModel=model.fit(train_images, train_lables, epochs=20)
        # model.save('D:\电池条带V2\model\my_model.h5')  # 保存为h5模型
        # tf.keras.models.save_model(model,"F:\python\moxing\model")#这样是pb模型
        # print("模型保存成功！")

        # history列表
        print(CnnModel.history.keys())
        font = {'family': 'Times New Roman','size': 12,}
        sns.set(font_scale=1.2)

        plt.plot(CnnModel.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('D:\\DCTDV2\\result\\V1\\loss' + "\\" + '%s.tif' % a,bbox_inches='tight',dpi=600)
        plt.show()

        plt.plot(CnnModel.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig('D:\\DCTDV2\\result\\V1\\accuracy' + "\\" + '%s.tif' % a,bbox_inches='tight',dpi=600)
        plt.show()