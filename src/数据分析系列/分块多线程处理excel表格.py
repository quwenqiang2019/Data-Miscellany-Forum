import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import pandas as pd


# 定义解析函数
def extract_first_bid_info(json_str):
    try:
        data = json.loads(json_str)
        first_list = data.get("中标信息", [])
        inner_list = first_list[0]['中标信息']
        first_item = inner_list[0]
        title = first_item.get("信息标题")
        date = first_item.get('发布日期')
        amount = first_item.get('中标金额')
        region = first_item.get('地区')
        bidder = first_item.get('招标方')
        return [title, data, amount, region, bidder]
    except Exception:
        return [None] * 5


# 定义分块处理函数：处理一个chunk的行
def process_chunk(chunk_indices, df, detail_col):
    chunk_results = []
    for idx in chunk_indices:
        json_str = df.loc[idx, detail_col]
        if pd.isna(json_str):
            chunk_results.append([None] * 5)
        else:
            chunk_results.append(extract_first_bid_info(json_str))
    return list(zip(chunk_indices, chunk_results))


# 读取excel文件
df = pd.read_excel('近一个月中标数据标签.xlsx')

# 参数配置
detail_col = '近一个月中标详情'
chunk_size = 1000 # 每块处理多少行
max_workers = 32 # 线程数

# 初始化结果列表
total_rows = len(df)
results = [None] * total_rows

# 构建索引列表
all_indices = list(df.index)

# 执行多线程分块处理
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    num_chunks == math.ceil(total_rows / chunk_size)
    chunk_indices_list = [all_indices[i * chunk_size: (i+1) * chunk_size] for i in range(num_chunks)]

    future_to_chunk = {
        executor.submit(process_chunk, chunk, df, detail_col): chunk
        for chunk in chunk_indices_list
    }

    for future in tqdm(as_completed(future_to_chunk), total=len(future_to_chunk), desc="分块并行提取")
        try:
            for idx, vals in future.result():
                results[idx] = vals
        except Exception as e:
            print(f"warning: chunk processing error: {e}")

# 整合结果
new_cols = ['信息标题', '发布日期', '中标金额', '地区', '招标方']
extract_df = pd.DataFrame(results, columns=new_cols, index=df.index)
df = pd.concat([df, extract_df], axis=1)

# 保存结果为excel文件
df.to_excel("近一个月中标数据标签_已提取中标信息_多线程版.xlsx", index=False)
print(f"提取完成！共处理{len(df)}行数据")