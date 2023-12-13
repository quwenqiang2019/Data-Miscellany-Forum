
l=['ab','bfff','ffffffff']
s=",".join(l)
print(s)#一个字符串
#将['ab','bfff','ffffffff']转化为('ab','bfff','ffffffff')
cmd = "("
for index, elem in enumerate(l):
    cmd = cmd + f"'{elem}'"
    if index < len(l) - 1:
        cmd += ','
cmd += ')'
print(cmd) #cmd仍然是一个字符串