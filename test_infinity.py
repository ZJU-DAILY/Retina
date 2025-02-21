# import infinity_embedded
import json
from datetime import datetime
import torch
import os 
import torch

#dense vector space size calculation

num_vectors = 670
vector_dimension = 768
vectors = torch.rand(num_vectors, vector_dimension, dtype=torch.float32)
print(vectors.shape)
file_path = '/data1/zhh/baselines/mm/icrr/random_vectors.pt'
torch.save(vectors, file_path)
file_size = os.path.getsize(file_path)

print(f"\nVectors have been saved to {file_path}")
file_size_MB = file_size / (1024 * 1024)
print(f"File size: {file_size_MB:.2f} MB")




#sparse_file space size calculation


import torch
import numpy as np
import pickle
from tqdm import tqdm
# 假设我们有如下参数设置
N = 150000  # 向量的维度
K = 200     # 每个向量的非零元素个数
num_vectors = 670 # 总向量数量

# 670000 
# 随机生成稀疏向量数据
indices = torch.randint(0, N, (2, K, num_vectors), dtype=torch.int64)  # 随机生成非零元素的索引
values = torch.rand(K, num_vectors).float()  # 随机生成非零元素的值

# 将索引展平为一维
indices_flat = indices.view(2, -1)  # 转换为 (2, K * num_vectors)
values_flat = values.view(-1)  # 转换为一维向量

# 使用稀疏矩阵表示向量
vectors = torch.sparse_coo_tensor(indices_flat, values_flat, torch.Size([N, num_vectors]))

inverted_index_ids = {}
inverted_index_floats = {}

for doc_id in tqdm(range(num_vectors), desc="Building Inverted Index"):
    indices_doc = vectors._indices()[:, doc_id].tolist()  # 获取非零元素索引
    values_doc = vectors._values().tolist()[doc_id * K: (doc_id + 1) * K]  # 获取非零元素值
    
    for term, value in zip(indices_doc, values_doc):
        if value > 0:  # 判断值是否大于0
            if term not in inverted_index_ids:
                inverted_index_ids[term] = []
                inverted_index_floats[term] = []

            inverted_index_ids[term].append(doc_id)
            inverted_index_floats[term].append(value)

file_path = "./inverted_index.pkl"
with open(file_path, "wb") as f:
    pickle.dump((inverted_index_ids, inverted_index_floats), f)

# 完成存储
print(f"倒排索引已保存到: {file_path}")

# Connect to infinity
# infinity_object = infinity_embedded.connect("/data1/zhh/baselines/mm/icrr/database")
# Retrieve a database object named default_db

# infinity_object.create_database("default_db")

# db_object = infinity_object.get_database("default_db")

# # Create a table with an integer column, a varchar column, and a dense vector column
# table_object = db_object.create_table("my_table", {"num": {"type": "integer"}, "body": {"type": "varchar"}, "vec": {"type": "vector, 4, float"}})
# # Insert two rows into the table
# table_object.insert([{"num": 1, "body": "unnecessary and harmful", "vec": [1.0, 1.2, 0.8, 0.9]}])
# table_object.insert([{"num": 2, "body": "Office for Harmful Blooms", "vec": [4.0, 4.2, 4.3, 4.5]}])
# # Conduct a dense vector search
# res = table_object.output(["*"]).match_dense("vec", [3.0, 2.8, 2.7, 3.1], "float", "ip", 2).to_pl()
# print(res)

# current_time = datetime.now().strftime("%Y-%m-%d-%H")
# print(current_time)
# json.dump({"timestamp": current_time}, f)