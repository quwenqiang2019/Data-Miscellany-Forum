import copy

original_list = [1, 2, [3, 4]]
deep_copied_list = copy.deepcopy(original_list)
# 修改深拷贝后的列表
deep_copied_list[2][0] = 5
print(original_list)  # [1, 2, [3, 4]]