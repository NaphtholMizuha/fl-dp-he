import numpy as np

# 创建一个 numpy 向量
data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

# 使用 bincount() 函数统计值的出现次数
counts = np.bincount(data)

# 获取排序的索引
sorted_indices = np.argsort(counts)

# 反转索引数组以实现降序排序
sorted_indices = sorted_indices[::-1]

# 输出排序后的值和它们的出现次数
for i in sorted_indices:
    if counts[i] > 0:
        print(f"Value: {i}, Count: {counts[i]}")