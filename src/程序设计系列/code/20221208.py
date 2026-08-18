import os, random, shutil

#======从文件夹中按比例随机选取图片移动到另一个文件夹=======
def moveFile_rate(fileDir):
    pathDir = os.listdir(fileDir)
    print(pathDir)
    for i in pathDir:
        filepath=fileDir+i
        filepathDir=os.listdir(filepath)
        print(filepathDir)
        filenumber=len(filepathDir)

        tarpath=tarDir +i
        print(tarpath)
        if not os.path.exists(tarpath):
            os.makedirs(tarDir +i)

        rate=0.5
        picknumber=int(filenumber*rate)
        sample = random.sample(filepathDir, picknumber)
        print(sample)
        for name in sample:
                shutil.move(filepath+'\\' +name, tarpath+'\\' +name)
    return

if __name__ == '__main__':
    fileDir ="D:\电池条带\案例\data\\"
    tarDir = 'D:\电池条带\案例\dataset\\train\\'
    moveFile_rate(fileDir)

