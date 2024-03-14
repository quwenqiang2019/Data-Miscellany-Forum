def count_lines(filename):
    with open(filename, 'r') as file:
        line_count = sum(1 for line in file)
    return line_count

filename = 'output1.txt'  # 将 'your_file.txt' 替换为您的txt文件路径

line_count = count_lines(filename)
print("Total number of lines in the file:", line_count)



