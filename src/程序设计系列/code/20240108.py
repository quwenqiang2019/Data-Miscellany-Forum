data = [1, 2, 3, 4, 5]

with open("output1.txt", "w") as file:
    for item in data:
        file.write(str(item) + "\n")




with open("output1.txt", "r") as file:
    data = file.read().splitlines()

print(data)
# processed_files = [int(item) for item in processed_files]

data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

with open("output2.txt", "w") as file:
    for row in data:
        for item in row:
            file.write(str(item) + ' ')
        file.write('\n')


with open('output2.txt', 'r') as file:
    lines = file.readlines()
    print(lines)
    data = []
    for line in lines:
        row = line.strip().split(' ')
        data.append(row)

print(data)