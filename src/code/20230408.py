
temp_str1 ='中国香港'

if temp_str1.find('中国大陆')!=-1:
    temp_str1=temp_str1.replace(temp_str1, 'Chinese')
elif temp_str1.find('中国香港')!=-1:
    temp_str1=temp_str1.replace(temp_str1, 'Hong Kong S.A.R.')

print(temp_str1)