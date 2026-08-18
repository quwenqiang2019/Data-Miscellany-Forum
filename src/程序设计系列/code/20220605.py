import os
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import os.path
import codecs
import pandas

#==========词料库构建=================
def Create_corpus(file):
    filePaths = []
    fileContents=[]
    for root, dirs, files in os.walk(file):
        print(root)
        print(dirs)
        print(files)
        # os.path.join()方法拼接文件名返回所有文件的路径，并储存在变量filePaths中
        for name in files:
            filePath=os.path.join(root, name)
            filePaths.append(filePath)
            print(filePaths)
            f = codecs.open(filePath, 'r', 'utf-8')
            print(f)
            fileContent = f.read()
            print(fileContent)
            f.close()
            fileContents.append(fileContent)
    #codecs.open()方法打开每个文件，用文件的read()方法依次读取其中的文本，将所有文本内容依次储存到变量fileContenst中，然后close()方法关闭文件。
    #创建数据框corpos，添加filePaths和fileContents两个变量作为数组
    corpos = pandas.DataFrame({'filePath': filePaths,'fileContent': fileContents})
    print(corpos)

Create_corpus("F:\医学大数据课题\AI_SLE\AI_SLE_TWO\TEST_DATA")
