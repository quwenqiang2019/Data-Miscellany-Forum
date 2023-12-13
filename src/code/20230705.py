list1 = [1, 2, 3, 4, 5]
list2 = [3, 4, 5, 6, 7]

set1 = set(list1)
set2 = set(list2)

diff1 = set1.symmetric_difference(set2)
print(diff1)
