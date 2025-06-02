#coding:utf-8
import os, json
import pandas as pd
import numpy as np
import sqlite3

# 使用ChatGPT生成的姓名，100个
names = ["王昕", "张柏宇", "张天宇", "李强", "高然", "杨洋", "马浩", "林昕", "陈林", "林涵", "孙宁", "徐坤", "丁旭", "王军", "丁宁", "孙琦", "高楠", "郭昱", "孙瑜", "赵磊", "李晨", "张亮", "孙文涛", "郭恒", "吴桦", "杨帆", "王乐轩", "杨铭", "高峰", "张伟", "郭锐", "林梓涵", "刘晓楠", "吴泽", "郭磊", "刘亮", "林泽", "林卓然", "林旭", "郭婷", "徐洋", "吴思源", "赵昀", "徐梓涵", "张健", "李俊熙", "丁乐", "李明哲", "吴涛", "郭宇飞", "马伟", "李翔", "赵阳", "陈浩天", "高子轩", "李峰", "孙尧", "陈敏", "吴强", "赵安然", "吴浩", "陈嘉豪", "刘昱", "杨宇", "孙冉", "张阳", "孙梅", "陈宁", "丁聪", "高翔", "孙家豪", "孙航", "周欢", "马丽", "马俊", "马羽", "赵丽", "周鑫", "李轩", "孙浩然", "周敏", "丁晓峰", "杨宇轩", "杨超凡", "徐乐", "刘强", "丁雪", "赵瑞", "周锐", "张阿颖", "孙浩", "孙桐", "周毅", "徐瑞", "吴迪", "王涛", "张宇轩", "高飞", "刘诚", "李娜"]
print(len(names))  # 输出100# 随机生成语数英物化生成绩，其中语数英满分150，物化生满分100

def random_score(total, number=100):
    return np.random.randint(0, total+1, size=number)

grades = {'姓名': names,"语文": random_score(150),"数学":random_score(150),"英语": random_score(150),"物理": random_score(100),"化学": random_score(100),"生物": random_score(100)}
df = pd.DataFrame(grades)
df['总分'] = df.iloc[:,1:].sum(axis=1)
print(df)
print(df.shape)


# 将dataframe转换为sqlite
conn = sqlite3.connect('database.db')
df.to_sql('grades', conn, if_exists='replace', index=False)
cursor = conn.cursor()
# 查询数据表
cursor.execute('''SELECT name FROM sqlite_master WHERE type='table' ''')
print(cursor.fetchall())
# 查询数据表的字段
cursor.execute('''SELECT * from sqlite_schema''')
print(cursor.fetchall())




def search_grades_by_name(name):
    '''     根据姓名精确查询该学生所有科目的成绩以及总分    '''
    resp = cursor.execute(f''' SELECT * from grades WHERE 姓名='{name}' ''')
    res = []
    for i in resp.fetchall():
        print(i)
        _name, chinese, math, en, phy, che, bio, total = i
        res.append(dict(姓名=_name, 语文=chinese, 数学=math, 英语=en, 物理=phy, 化学=che, 生物=bio, 总分=total))
    return res

def search_grades_by_lastname(lastname):
    '''     根据姓氏查询该姓氏学生的所有科目成绩以及总分    '''
    resp = cursor.execute(f''' SELECT * from grades WHERE 姓名 LIKE '{lastname}%' ''')
    res = []
    for i in resp.fetchall():
        _name, chinese, math, en, phy, che, bio, total = i
        res.append(dict(姓名=_name, 语文=chinese, 数学=math, 英语=en, 物理=phy, 化学=che, 生物=bio, 总分=total))
    return res

def search_by_name_subject(name, subject):
    '''     根据学生姓名和科目查询某科目成绩或者总分    '''
    resp1 = search_grades_by_name(name)
    res = []
    for i in resp1:
        res.append({'姓名': i['姓名'], f'{subject}': i[subject]})
    return res

def filter_by_subject_score(subject, score):
    '''     根据某科目分数大于某阈值筛选学生姓名和该科目对应分数    '''
    resp = cursor.execute(f''' SELECT * from grades WHERE '{subject}' >= '{score}' ''')
    res = []
    for i in resp.fetchall():
        _name, chinese, math, en, phy, che, bio, total = i
        res.append(dict(姓名=_name, 语文=chinese, 数学=math, 英语=en, 物理=phy, 化学=che, 生物=bio, 总分=total))
    return res

def filter_by_total_score_top_n(subject, top_n=5):
    '''     根据某科目或者总成绩排名，并获取前N名学生的成绩信息    '''
    resp = cursor.execute(f''' SELECT * from grades ORDER BY {subject} DESC LIMIT {top_n}  ''')
    res = []
    for i in resp.fetchall():
        _name, chinese, math, en, phy, che, bio, total = i
        res.append(dict(姓名=_name, 语文=chinese, 数学=math, 英语=en, 物理=phy, 化学=che, 生物=bio, 总分=total))
    return res

# 测试这些函数调用没有问题
print(search_grades_by_name('张天宇'))
print(search_grades_by_lastname('高'))
print(search_by_name_subject('张天宇', '英语'))
print(filter_by_subject_score('语文', 120))
print(filter_by_total_score_top_n('总分', 5))



tools = [
    {
        "type": "function",
        "function":{"name": "search_grades_by_name",
                    "description": "根据姓名精确查询该学生所有科目的成绩以及总分",
                    "parameters":{"type": 'object',
                                  "properties":{ "name": { "type": "string", "description": "学生姓名" }},
                                  "required": ["name"]
                                  },
                    }
    },
    {
        "type": "function",
        "function":{"name": "search_grades_by_lastname",
                    "description": "根据姓氏查询该姓氏学生的所有科目成绩以及总分",
                    "parameters":{"type": 'object',
                                  "properties":{"lastname": {"type": "string", "description": "学生的姓氏" }   },
                                  "required": ["lastname"]
                                  },
                    }
    },
    {
        "type": "function",
        "function":{ "name": "search_by_name_subject",
                     "description": "根据学生姓名和科目查询某科目成绩或者总分",
                     "parameters":{"type": 'object',
                                   "properties":{ "name": {"type": "string","description": "学生姓名"},
                                                  "subject":{ "type": "string",  "description": "科目或者总分"}},
                                   "required": ["name", "subject"]
                                   },
                     }
    },
    {
        "type": "function",
        "function":{"name": "filter_by_subject_score",
                    "description": "根据某科目分数大于某阈值筛选学生姓名和该科目对应分数",
                    "parameters":{"type": 'object',
                                  "properties":{"score": { "type": "int", "description": "成绩分数" },
                                                "subject":{  "type": "string", "description": "科目或者总分"}  },
                                  "required": ["score", "subject"]
                                  },
                    }
    },
    {
        "type": "function",
        "function":{ "name": "filter_by_total_score_top_n",
                     "description": "根据学生姓名和科目查询某科目成绩或者总分",
                     "parameters":{ "type": 'object',
                                    "properties":{ "top_n": {  "type": "int", "description": "排名前几"},
                                                   "subject":{ "type": "string",   "description": "科目或者总分"          }                },
                                    "required": ["subject"]
                                    },
                     }
    },
]

print(len(tools))


import json
from zhipuai import ZhipuAI
api_key = 'your_api_key'
api_key = '6358a059f55445ec800a5c3554a0d686.3eXwmE2Il7C6yr6Q'
client = ZhipuAI(api_key=api_key)
system_prompt = """你是一位专业的 AI 助手，你的任务是回答用户问题，可以利用工具接口获得用户想要的答案"""
def call_glm(messages, model="glm-4-plus", temperature=0.95, tools=None, top_p=0.7):
    response = client.chat.completions.create(model=model,
                                              messages=messages,
                                              temperature=temperature,
                                              top_p=top_p,
                                              tools=tools)
    return response

query = "刘晓楠的语文成绩？"
messages = [ {"role": "system", "content": system_prompt},
             {"role": "user", "content": query}    ]
try:
    response = call_glm(messages, tools=tools)
    messages.append(response.choices[0].message.model_dump())
except Exception as e:
    print(e)

tools_call = response.choices[0].message.tool_calls[0]
print(tools_call.function)
tool_name = tools_call.function.name
args = tools_call.function.arguments
print(tool_name, args)


# 调用函数，获取函数返回结果，然后再使用LLM进行问答
fc_res = json.dumps(eval(tool_name)(**json.loads(args)))  # function calling result
messages.append({"role": "tool",  "content": f"{fc_res}", "tool_id": tools_call.id})
try:
    response = call_glm(messages, tools=tools)
    messages.append(response.choices[0].message.model_dump())
except Exception as e:
    print(e)
print(messages[-1])

