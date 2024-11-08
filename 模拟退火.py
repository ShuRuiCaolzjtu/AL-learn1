import math
import copy
import numpy as np
import random
import matplotlib.pyplot as plt

# 城市坐标  20个
cities = np.array([
    [60, 200], [180, 200], [80, 180], [140, 180],
    [20, 160], [100, 160], [200, 160], [140, 140],
    [40, 120], [100, 120], [180, 100], [60, 80],
    [120, 80], [180, 60], [20, 40], [100, 40],
    [200, 40], [20, 20], [60, 20], [160, 20]
])

# 计算距离矩阵
def compute_distance_matrix(cities):#计算每两个城市之间的欧氏距离
    num_cities = len(cities)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])
    return distance_matrix

distance_matrix = compute_distance_matrix(cities)

# 计算路径的总距离
def path_distance(path, distance_matrix):#计算path路径下的总路程
    path = [p - 1 for p in path]  # 将路径索引减1以匹配城市索引
    distance = 0
    for i in range(len(path) - 1):
        distance += distance_matrix[path[i]][path[i + 1]]
    distance += distance_matrix[path[-1]][path[0]]  # 回到起点
    return distance

# 模拟退火算法

# 随机生成初始解
def Get_initial_solution(length, lower_bound, upper_bound):
    return random.sample(range(lower_bound, upper_bound + 1), length)

#选择初始路径中三个城市并随机交换顺序
def Get_test_path(path,n):
    path_copy=copy.deepcopy(path)
    exchange_index=random.sample(range(0, 20), 2*n)
    exchange_index1 = exchange_index[:n]#替换的n个城市索引
    exchange_index2 = exchange_index[n:]#被替换的n个城市索引

    for i in range(len(exchange_index1)):
        path_copy[exchange_index1[i]]=path[exchange_index2[i]]
        path_copy[exchange_index2[i]]=path[exchange_index1[i]]

    return path_copy



# 随机生成初始的旅行顺序
path = Get_initial_solution(20, 1, 20)
print(path)

#计算初始值
initial_distance = path_distance(path, distance_matrix)
distance=initial_distance

#模拟退火循环
now_iteration=1#初始化迭代次数
max_iteration=100#最大迭代次数
temperature=1000000#初始温度
min_temperature=0.1#最小温度
path_list=[path]
distance_list=[distance]

while now_iteration<=max_iteration and temperature>min_temperature:
    path_update = copy.deepcopy(path)
    # 进行路径解的邻域搜索 每次替换三个解元素
    path_update = Get_test_path(path_update, 3)
    # 计算邻域搜索后的值
    distance_update = path_distance(path_update, distance_matrix)
    # 模拟退火判断是否接受
    change_update=0#初始化概率
    if distance_update<distance:
        change_update=1
    else:change_update=math.e**((distance-distance_update)/(1*temperature))#模拟退火核心函数
    #随机生成一个0-1之间的数代表接受的可能性
    update_value = random.random()
    if update_value<=change_update:
        path=path_update
        distance=distance_update
    else:
        path=path
        distance=distance
    #记录结果
    path_list.append(path)
    distance_list.append(distance)
    #更新温度与迭代次数
    print("迭代次数：{}".format(now_iteration))
    print("当前温度：{}".format(temperature))
    print("当前最优路径：{}".format(path))
    print("当前最优值：{}".format(distance))
    now_iteration=now_iteration+1
    temperature=0.75*temperature
    #播报结果



# 提取最优解
best_path = path_list[-1]
best_distance = distance_list[-1]


#可视化最优质
# 设置 x 轴的索引
x_values = range(len(distance_list))
# 绘制折线图
plt.plot(x_values, distance_list, marker='o')
# 添加标题和标签
plt.title("Optimal value iteration process")
plt.xlabel("Number of iterations")
plt.ylabel("value")
# 显示图形
plt.show()

# best_path=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
# 可视化最优路径
plt.figure(figsize=(8, 6))
plt.scatter(cities[:, 0], cities[:, 1], c='red', marker='o')
for i in range(len(best_path) - 1):
    plt.plot([cities[best_path[i] - 1][0], cities[best_path[i + 1] - 1][0]],
             [cities[best_path[i] - 1][1], cities[best_path[i + 1] - 1][1]],
             'b-')
plt.plot([cities[best_path[-1] - 1][0], cities[best_path[0] - 1][0]],
         [cities[best_path[-1] - 1][1], cities[best_path[0] - 1][1]],
         'b-')
# plt.title(f"Best Path with Distance: 0")
plt.title(f"Best Path with Distance: {best_distance:.2f}")
plt.show()

