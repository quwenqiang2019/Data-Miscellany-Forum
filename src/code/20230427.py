def query_batch(self, engine, batch_step, end, sql):
    session = make_session(engine)
    cursor = session.execute(sql.format(batch_step, end))
    fields = cursor._metadata.keys
    df = pd.DataFrame([dict(zip(fields, item)) for item in cursor.fetchall()])
    print(len(df))
    cursor.close()
    session.close()
    return df


# 批量查询并做表连接处理
sql1 = 'select count(*) from clusterAllReplicas (clickhouse, data_mgmt.gha_activity_2023_03)'
start = 0
max_length = make_session(engine).execute(sql1).fetchall()[0][0]
batch_step = 5000000

result_df = pd.DataFrame()
for i in range(start, max_length, batch_step):
    print(i)
    print(batch_step, i)
    df2 = query_batch(engine, batch_step, i,
                      sql='select * from clusterAllReplicas (clickhouse, data_mgmt.gha_activity_2023_03) order by  id limit {} offset {}')

    # df2['repo_url'] = df2['repo.url'].map(lambda x: Object2.string_extraction(x))
    temp_df = merge_data(df1, df2)
    if len(temp_df):
        result_df = result_df.append(temp_df)

print(len(result_df))
result_df.to_excel('D:\工作\ospp-report\\result\%d.xlsx' % i)