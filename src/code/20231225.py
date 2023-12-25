import argparse


def parseArgs():
    # 创建一个ArgumentParser对象
    parser = argparse.ArgumentParser(description='这是一个命令行参数解析的示例程序')
    # 添加一个位置参数
    parser.add_argument('name', type=str, help='你的名字')
    # 添加一个可选参数
    parser.add_argument('--age', '-a', type=int, default=18, help='你的年龄')
    # 解析命令行输入
    args = parser.parse_args()

    return args

def task(name, age):
    print('你好，{}！你的年龄是{}'.format(name, age))

def main():
    args = parseArgs()
    task(args.name, args.age)


if __name__  == '__main__':
    main()