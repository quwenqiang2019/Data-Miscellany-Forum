import re
import ast

x1 = str([None,'master','hh'])
print(x1)

x2 = ast.literal_eval(x1)
print(x2)

x3 = ",".join(str(item) for item in x2)
print(x3)

s=re.search(r'(main|master)', x3)
print(s)
