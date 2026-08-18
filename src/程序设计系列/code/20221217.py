import cv2
import numpy as np
from cv2 import dnn_superres
import time
import os, random, shutil
from PIL import Image

#==================读取图片=================================
def read_image(paths):
    os.listdir(paths)
    filelist = []
    for root, dirs, files in os.walk(paths):
        for file in files:
            if os.path.splitext(file)[1] == ".png":
                filelist.append(os.path.join(root, file))

    print(filelist)
    return filelist

#==========转换图片像素，使其大小一致===============
def im_xiangsu_1(paths,path):
    for filename in paths:
        print(filename)
        print(filename[29:-4])
        try:
            im = Image.open(filename)
            newim = im.resize((128, 128))
            print(newim)
            newim.save(path+'\\'+filename[29:-4] + '.png')
            print('图片' + filename[29:-4] + '.png' + '像素转化完成')
        except OSError as e:
            print(e.args)

if __name__ == '__main__':
    tarDir_Train = 'D:\电池条带V2\\dataset\\train\\'    #移动到新的文件夹路径
    pathDir_Train = os.listdir(tarDir_Train)
    print(pathDir_Train)
    for i in pathDir_Train:                #转换图片像素，使其大小一致
        im_xiangsu_1(read_image(tarDir_Train+i), tarDir_Train+i)