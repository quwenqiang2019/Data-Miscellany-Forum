a = ['a', 'b', 'c', 'd']
b = ['1', '2', '3', '4']

res = list(zip(a, b))
print(res)


origin = zip(*res)  # #前面加*号，事实上*号也是一个特殊的运算符，叫解包运算符
print(list(origin))

s = ["flower","flow","flight"]
print(list(zip(*s)))


s = "A man, a plan, a canal: Panama"
l = filter(str.isalnum, s.lower())
# print(list(l))
s1 = ''.join(l)
print(s1)
print(s1 == s1[::-1])


s = "paper"
t = "title"

from collections import Counter

dicts = Counter(s)
dictt = Counter(t)

for i in range(len(s)):
    print(list(dicts.keys()))
    print(s[i])
    inds = list(dicts.keys()).index(s[i])
    print(inds)

