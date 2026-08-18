# 对列表进行排序：
numbers = [5, 2, 8, 1, 9, 3]
sorted_numbers = sorted(numbers)
print(sorted_numbers)
# 输出: [1, 2, 3, 5, 8, 9]

# 使用关键字函数进行排序
students = [
    {'name': 'Alice', 'age': 20},
    {'name': 'Bob', 'age': 18},
    {'name': 'Charlie', 'age': 22}
]
sorted_students = sorted(students, key=lambda x: x['age'])
print(sorted_students)
# 输出: [{'name': 'Bob', 'age': 18}, {'name': 'Alice', 'age': 20}, {'name': 'Charlie', 'age': 22}]