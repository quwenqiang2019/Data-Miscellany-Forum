# 读取txt文件，以二维列表形式输出，每一个元素为一行
file=open('G:\数据杂坛\素材\\1120\文本.txt',mode='r',encoding='UTF-8')
admin=[]
# 读取所有行(直到结束符 EOF)并返回列表
contents = file.readlines()
print(contents)
for msg in contents:
    # 删除结尾的\n字符
    msg = msg.strip('\n')
    # 字符串根据空格进行分割
    adm = msg.split(' ')
    admin.append(adm)
file.close()
print(admin)


# 将二维列表写入txt文件，每一个元素一行
with open('G:\数据杂坛\素材\\1120\文本副本.txt','w') as f:
    for i in admin:
        for j in i:
            f.write(j)
            f.write(' ')
        f.write('\n')
    f.close()

