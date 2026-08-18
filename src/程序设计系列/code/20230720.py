import re

string = "Linux Python Cloud Native Distributed System AI C++ Deep Learning Framework Micro Service Automation Git IoT"

# 定义要保留的特定词组
special_phrases = ["Deep Learning Framework", "Micro Service", "Distributed System"]

# 将特定词组替换为占位符
for phrase in special_phrases:
    string = re.sub(r'\b' + re.escape(phrase) + r'\b', f'#{special_phrases.index(phrase)}#', string)

# 替换剩余的空格为逗号
modified_string = string.replace(' ', ',')

# 将占位符还原为特定词组
for i, phrase in enumerate(special_phrases):
    modified_string = modified_string.replace(f'#{i}#', phrase)

print(modified_string)