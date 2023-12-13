# 方法一：基于clickhouse_connect
import clickhouse_connect

client = clickhouse_connect.get_client(host="你的服务器地址", port='你的port', database='你的db',user="你的用户名" ,password="你的password")
sql = 'select * from data_mgmt.gha_activity_2023_02'
#res = client.command(sql)
res = client.query(sql)
print(res.result_set)


# 方法二：基于sqlalchemy
from sqlalchemy import create_engine
import pandas as pd
from clickhouse_sqlalchemy import make_session

conf = {
    "server_host": '你的服务器地址',
    "port": '输入你的port',
    "user": '你的用户名',
    "password": '输入你的password',
    "db": '输入你的db'
}
connection = 'clickhouse://{user}:{password}@{server_host}:{port}/{db}'.format(**conf)
engine = create_engine(connection)
sql = 'select * from data_mgmt.gha_activity_2023_02'

# 2.1
df=pd.read_sql_query(sql,engine)

# 2.2
session = make_session(engine)
cursor = session.execute(sql)
try:
    fields = cursor._metadata.keys
    df = pd.DataFrame([dict(zip(fields, item)) for item in cursor.fetchall()])
finally:
    cursor.close()
    session.close()

