import os
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import os.path
import codecs
import pandas
import jieba

#==========词料库构建===============
def Create_corpus(file):
    filePaths = []
    fileContents=[]
    for root, dirs, files in os.walk(file):
        # os.path.join()方法拼接文件名返回所有文件的路径，并储存在变量filePaths中
        for name in files:
            filePath=os.path.join(root, name)
            filePaths.append(filePath)
            f = codecs.open(filePath, 'r', 'utf-8')
            fileContent = f.read()
            f.close()
            fileContents.append(fileContent)
    #codecs.open()方法打开每个文件，用文件的read()方法依次读取其中的文本，将所有文本内容依次储存到变量fileContenst中，然后close()方法关闭文件。
    #创建数据框corpos，添加filePaths和fileContents两个变量作为数组
    corpos = pandas.DataFrame({'filePath': filePaths,'fileContent': fileContents})
    return corpos

#============中文分词===============
def Word_segmentation(corpos):
    segments = []
    filePaths = []
    #遍历语料库的每一行数据，得到的row为一个个Series，index为key
    for index, row in corpos.iterrows():
        print(row)
        filePath = row['filePath']#获取每一个row中filePath对应的文件路径
        fileContent = row['fileContent']#获取row中fileContent对应的每一个文本内容
        segs = jieba.cut(fileContent)#对文本进行分词
        for seg in segs:
            segments.append(seg)#分词结果保存到变量segments中
            filePaths.append(filePath)#对应的文件路径保存到变量filepaths中
    #将分词结果及对应文件路径添加到数据框中
    segmentDataFrame = pandas.DataFrame({'segment': segments,'filePath': filePaths})
    print(segmentDataFrame)
    return

corpos=Create_corpus("F:\医学大数据课题\AI_SLE\AI_SLE_TWO\TEST_DATA")
Word_segmentation(corpos)
