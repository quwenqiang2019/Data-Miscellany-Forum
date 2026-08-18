import cv2
import numpy as np
from cv2 import dnn_superres
import time
import os, random, shutil
from PIL import Image
#===============从文件夹中按数量随机选取一定数量图片移动到另一个文件夹====================
def Superresolution(fileDir,tarDir):
    pathDir = os.listdir(fileDir)
    print(pathDir)
    del pathDir[0:4]
    print(pathDir)

    for i in pathDir:
        filepath=fileDir+'\\'+i
        print(filepath)
        filepathDir=os.listdir(filepath)
        print(filepathDir)
        for a in filepathDir:
            print(a)
            print(filepath+'\\'+a)
            # 创建SR对象...
            sr = dnn_superres.DnnSuperResImpl_create()
            # 读图
            input = cv2.imread(filepath+'\\'+a)
            # 读取模型
            sr.readModel("D:\DCTDV2\model\EDSR_x4.pb")
            # 设定算法和放大比例
            sr.setModel("edsr", 4)
            # 将图片加载入模型处理，获得超清晰度图片
            print("处理图片中...\n")
            t0 = time.perf_counter()
            upScalePic = sr.upsample(input)
            print("处理图片完成\n")
            print(time.perf_counter() - t0)
            # 输出
            tarpath = tarDir+'\\'+i
            print(tarpath)
            if not os.path.exists(tarpath):  # 判断文件夹是否已经存在
                os.makedirs(tarpath)
            print(tarpath)
            cv2.imwrite(tarpath+'\\'+a, upScalePic)
            print("输出图片完成\n")
    return

if __name__ == '__main__':
    fileDir ="D:\DCTDV2\dataset\\train"  #源图片文件夹路径
    tarDir="D:\DCTDV2\dataset\\trainSR"
    Superresolution(fileDir,tarDir)