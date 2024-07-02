a = ['a', 'b', 'c', 'd']
b = ['1', '2', '3', '4']

res = list(zip(a, b))
print(res)


origin = zip(*res)  # #前面加*号，事实上*号也是一个特殊的运算符，叫解包运算符
print(list(origin))

s = ["flower","flow","flight"]
print(list(zip(*s)))


dic = {'a':1, 'b':2, 'c':3}
print({**dic})

#
#
# s = "A man, a plan, a canal: Panama"
# l = filter(str.isalnum, s.lower())
# # print(list(l))
# s1 = ''.join(l)
# print(s1)
# print(s1 == s1[::-1])
#
#
# s = "paper"
# t = "title"
#
# from collections import Counter
#
# dicts = Counter(s)
# dictt = Counter(t)
#
# for i in range(len(s)):
#     print(list(dicts.keys()))
#     print(s[i])
#     inds = list(dicts.keys()).index(s[i])
#     print(inds)
#
#
#
# def removeDuplicates(nums: list[int]) -> int:
#     slow, fast = 0, 1
#     while fast < len(nums):
#         if nums[fast] != nums[slow]:
#             slow += 1
#             nums[slow] = nums[fast]
#         fast += 1
#
#     return slow+1
#
#     # nums = list(set(nums))
#     #
#     # print(nums)
#
#     return nums
#
#
# nums = [1, 1, 2]
# removeDuplicates(nums)


# def twoSum(nums: list[int], target: int) -> list[int]:
#     hashtable = dict()
#     for i, num in enumerate(nums):
#         print(hashtable)
#         if target - num in hashtable:
#             print(hashtable[target - num])
#             return [hashtable[target - num], i]
#         hashtable[num] = i
#     return []
#
# twoSum([2,10,7,11,15], 9)


# def romanToInt(s: str) -> int:
#     hashtable = {
#         "I": 1,
#         "V": 5,
#         "X": 10,
#         "L": 50,
#         "C": 100,
#         "D": 500,
#         "M": 1000,
#         "a": 4,
#         "b": 9,
#         "c": 40,
#         "d": 90,
#         "e": 400,
#         "f": 900
#     }
#
#     res = 0
#     s = s.replace("IV", "a")
#     s = s.replace("IX", "b")
#     s = s.replace("XL", "c")
#     s = s.replace("XC", "d")
#     s = s.replace("CD", "e")
#     s = s.replace("CM", "f")
#
#     for i, v in enumerate(s):
#         temp = hashtable[v]
#         res = res + temp
#
#     return res
#
# romanToInt("III")