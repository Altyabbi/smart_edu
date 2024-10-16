import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 问题总结：已知知识点网络T，评估学习者知识状态K，然后求出（个性化）学习路径P
# Step0：前置
# 已知知识点网络T转换为对应的邻接矩阵A
D = np.array(['a', 'b', 'c', 'd', 'e'])  # 知识领域D + 约束关系R = 知识点网络T
# example0：
A = np.array([[0, 1, 1, 0, 0, ],
              [0, 0, 0, 1, 0, ],
              [0, 0, 0, 0, 1, ],
              [0, 0, 0, 0, 0, ],
              [0, 0, 0, 0, 0, ], ])
# # example1：
# A = np.array([[0,1,1,0,0],
#               [0,0,0,0,0],
#               [0,0,0,1,1],
#               [0,0,0,0,0],
#               [0,0,0,0,0], ])
# # example2：
# A = np.array([[0,1,1,1,0],
#               [0,0,0,0,1],
#               [0,0,0,0,1],
#               [0,0,0,0,1],
#               [0,0,0,0,0], ])
print("A:", A)

# Step1：计算可达矩阵R
def calBoolMul(X, Y):
    """
    计算X⊗Y
    """
    # Bool And
    # return X & Y  # [True, False] -> [True, False]
    # return np.logical_and(X, Y)  # [0, 1] -> [True, False]
    # return np.logical_and(X, Y).astype(int)  # [0, 1] -> [0, 1]

    # Boolean Mul
    # return np.dot(X, Y)  # [0, 1] -> [int, int]
    return (np.dot(X, Y) > 0).astype(int)  # [0, 1] -> [0, 1]


n = A.shape[0]
E = np.eye(n)  # 创建单位矩阵（与邻接矩阵A同形状）
R = A  # 可达矩阵R初始化为邻接矩阵A
AplusE_k = A+E  # (A+E)^k初始化为(A+E)^1
for k in range(n):
    AplusE_kplus1 = calBoolMul(AplusE_k, A+E)
    # print(AplusE_kplus1)
    if np.array_equal(AplusE_k, AplusE_kplus1):
        R = AplusE_k
        break
    AplusE_k = AplusE_kplus1  # 更新(A+E)^k
print("R：", R)


# # Step2：计算Qb矩阵(所有有效知识点组合)
# # print("R1:", R[:, 0], type(R[:, 0]))
# a = np.array([R[:, 0]]).T
# # print(a)
# Qb = a
# # print(Qb)
# m = 1
# for j in range(1,n):  # [1,n-1]  ↑
#     for t in range(m-1,-1,-1):  # [m-1, 0] ↓
#         b = np.logical_or(R[:,j], a[:,t]).astype(int)
#         # print(R[:,j], a[:,t], b)
#         # print(b)
#
#         # print(np.any(np.all(Qb == b, axis=0)))
#         # print(np.all(np.all((Qb == b[:, np.newaxis]) == False, axis=0)))
#         # if not np.any(np.all(Qb == b, axis=0)):  # b与Qb任意一列均不相同
#
#         flag = True
#         for i in range(Qb.shape[1]):
#             # print(Qb[:, i], b)
#             if np.array_equal(Qb[:, i], b):
#                 flag = False
#                 break
#         if flag:  # b与Qb任意一列均不相同
#             # print(a)
#             # print(b)
#             # 此处不确定 am=b
#             # print(m, a.shape[1]-1)
#             if m == a.shape[1]:
#                 a = np.hstack((a, np.array([b]).T))
#             else:
#                 a[:, m] = b
#             # if m < a.shape[1]:
#             #     a[:, m] = b
#             # else:
#             #     a = np.hstack((a, np.array([b]).T))
#             # print(a)
#             Qb = np.hstack((Qb, np.array([b]).T))  # 并把am(b)放入Qb中
#             m += 1
#         # break
# # print(np.any(np.all(R == a, axis=0)))
# print(Qb)
# #
# Qb1 =
# 1 1 1 1 1 1 1 1 1
# 0 1 0 1 0 1 1 1 1
# 0 0 1 0 1 1 1 1 1
# 0 0 0 1 0 0 0 1 1
# 0 0 0 0 1 0 1 0 1


# Step2：计算Qb矩阵(所有有效知识点组合)
# print("R1:", R[:, 0], type(R[:, 0]))
# a = np.array([R[:, 0]]).T
# print(a)
# Qb = True
# print(Qb)

# # Qb = np.array([R[:, 0]]).T
# m = 0
# for j in range(0,n):  # [0,n-1]  ↑
#     m += 1
#     if 'Qb' not in globals():
#         print("not defined!")
#         Qb = np.array([R[:,j]]).T
#     else:
#         Qb = np.hstack((Qb, np.array([R[:,j]]).T))
#     # Qb = np.hstack((Qb, np.array([R[:, j]]).T))
#
#     for t in range(m-1,-1,-1):  # [m-1, 0] ↓
#         # print(np.logical_or(R[:,j], Qb[:,t]))
#         b = np.logical_or(R[:,j], Qb[:,t]).astype(int)
#         print(b)
#
#         flag = True
#         # print(Qb.shape[1])
#         for i in range(Qb.shape[1]):
#             # print(Qb[:, i], b)
#             # print(np.array_equal(Qb[:, i], b))
#             if np.array_equal(Qb[:, i], b):
#                 flag = False
#                 break
#         if flag:  # b与Qb任意一列均不相同
#             print(Qb)
#             m += 1
#
#             while(m >= Qb.shape[1]):  # 凑齐Qb矩阵
#                 print("11111")
#                 Qb = np.hstack((Qb, np.zeros((Qb.shape[0], 1))))
#             # print(m, Qb.shape[1])
#             Qb[:, m] = np.array([b]).T[0]
#
#             # if m == Qb.shape[1]:
#             #     Qb = np.hstack((Qb, np.array([b]).T))
#             # else:
#             #     Qb[:, m] = np.array([b]).T[0]
#
#             # Qb = np.hstack((Qb, np.array([b]).T))  # 并把am(b)放入Qb中
#             # m += 1
#         # break
# # print(np.any(np.all(R == a, axis=0)))
# print(Qb)
# #

# Qb = np.array([R[:, 0]]).T  # Qb初值
# m = 0
# for j in range(0, n):  # [0, n-1]，共n次循环
#     m += 1
#     # print(R[:, j])
#     # print(np.array([R[:, j]]).T)
#     # Qb = np.hstack((Qb, np.array([R[:, j]]).T))  # 到此处肯定是新增一列  Qb[:, m] = R[:, j]
#     # print(Qb)
#
#     if 'Qb' not in globals():
#         # print("not defined!")
#         Qb = np.array([R[:,j]]).T
#     else:
#         Qb = np.hstack((Qb, np.array([R[:, j]]).T))  # Qb[:, m-1] = ...  （下标要减1
#
#     for t in range(m-2,-1,-1):  # [m-2, 0] 共m-1次
#         b = np.logical_or(R[:, j], Qb[:, t]).astype(int)
#         # print(b)
#
#         flag = True
#         for i in range(Qb.shape[1]):
#             if np.array_equal(Qb[:, i], b):
#                 flag = False
#                 break
#         if flag:  # b与Qb任意一列均不相同
#             m += 1
#             Qb = np.hstack((Qb, np.array([b]).T))  # Qb[:, m-1] = ...  （下标要减1
# print(Qb)

# # 此处令Qb等于论文原本的结果（论文中可能排了个序？）
# Qb = np.array([[1,1,1,1,1,1,1,1,1],
#                [0,1,0,1,0,1,1,1,1],
#                [0,0,1,0,1,1,1,1,1],
#                [0,0,0,1,0,0,0,1,1],
#                [0,0,0,0,1,0,1,0,1],])
# print(Qb)

# 这一步论文的时间复杂度分析有误！

a = np.array([R[:, 0]]).T
Qb = a
m = 2
for j in range(1, n):  # [1, n-1]，共n-1次循环
    for t in range(m-2,-1,-1):  # [m-2, 0] 共m-1次
        b = np.logical_or(R[:, j], a[:, t]).astype(int)
        # print(b)

        flag = True
        for i in range(Qb.shape[1]):
            if np.array_equal(Qb[:, i], b):
                flag = False
                break
        if flag:  # b与Qb任意一列均不相同
            a = np.hstack((a, np.array([b]).T))
            Qb = np.hstack((Qb, np.array([b]).T))  # Qb[:, m-1] = ...  （下标要减1
            m += 1
print(Qb)




letters = [chr(ord('a') + i) for i in range(Qb.shape[0])]
# print(letters)
mp = {i: chr(ord('a') + i) for i in range(Qb.shape[0])}  # {0:'a' .... Qb.shape[0]-1:'xxx'}

# print(mp)
def diamond(M):  # 与M中问题相关的知识点组合
    """
    :param M: q
    :return: 知识点组合
    """
    z = []
    for idx in range(len(M)):
        if M[idx] == 1:
            z.append(mp[idx])
    return z

def square(N):  # 仅 与N中所有知识点相关的问题集
    """
    :param N: T（知识点组合）
    :return: 问题集
    """
    q = []
    for col in range(Qb.shape[1]):
        diamond_q = diamond(Qb[:, col])
        if set(diamond_q) <= set(N):  # dia_q中元素是否均在N中
            q.append(col+1)
    return q
    # for x in N:
    #     # map(chr, range(ord('a'), ord('a') + len(mp.keys())))
    #     z2q = [1 if c in x else 0 for c in map(chr, range(ord('a'), ord('a') + len(mp.keys())))]
    #     q.append(z2q)
    # return q

# print(diamond(Qb[:, 5]))
# print(square(diamond(Qb[:, 5])))

Z = [[]]
for col in range(Qb.shape[1]):  # 遍历Qb每一列
    Z.append(diamond(Qb[:, col]))
    # z = []
    # for row in range(Qb.shape[0]):
    #     if Qb[row][col] == 1:
    #         z.append(mp[row])
    # Z.append(z)
print(Z)



# 定义输入数组
sorted_Z = sorted(Z, key=lambda x: len(x))
print(sorted_Z)
tuple_sorted_Z = [tuple(e) if e else () for e in sorted_Z]
# elements = [(), ('a',), ('a', 'b'), ('a', 'c'), ('a', 'b', 'd'),
#             ('a', 'c', 'e'), ('a', 'b', 'c'), ('a', 'b', 'c', 'e'),
#             ('a', 'b', 'c', 'd'), ('a', 'b', 'c', 'd', 'e')]

# 创建有向图
G = nx.DiGraph()

# 遍历每个元素
for i, elem1 in enumerate(tuple_sorted_Z):
    # 遍历其他元素，寻找比当前元素多一个字母的元素
    for j, elem2 in enumerate(tuple_sorted_Z[i+1:], start=i+1):
        if len(elem2) == len(elem1) + 1 and set(elem1).issubset(set(elem2)):
            added_letter = set(elem2) - set(elem1)  # 找到新增的字母
            added_letter = next(iter(added_letter))  # 将集合转换为单个字符
            # print(elem1, type(elem1))
            # G.add_edge(elem1, elem2, label=added_letter)
            G.add_edge(tuple(square(list(elem1))), tuple(square(list(elem2))), label=added_letter)


# # 打印图的节点和边
# print("Nodes:", G.nodes())
# print("Edges:")
# for edge in G.edges(data=True):
#     print(edge)

# # 如果需要可视化，可以使用下面的代码
# nx.draw(G, with_labels=True, node_size=1000, node_color="skyblue", font_size=10, font_color="black")
# plt.show()
# 按照节点的元素个数排序
sorted_nodes = sorted(G.nodes(), key=lambda x: len(x))

# 绘制图形
pos = nx.spring_layout(G, seed=2025)  # 设置随机种子，使得每次绘制的结果相同  example00
# pos = nx.spring_layout(G, seed=2024)  # 设置随机种子，使得每次绘制的结果相同  example01

# pos = nx.spring_layout(G, seed=2022)  # 设置随机种子，使得每次绘制的结果相同  example1
#
# pos = nx.spring_layout(G, seed=2022)  # 设置随机种子，使得每次绘制的结果相同  example2

nx.draw(G, pos, with_labels=True, node_size=1500, node_color="skyblue", font_size=10, font_color="black")
labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

# 最短路径
start_node = tuple(square(sorted_Z[0]))
end_node = tuple(square(sorted_Z[-1]))
all_shortest_paths = nx.all_shortest_paths(G, source=start_node, target=end_node)
for path in all_shortest_paths:
    shortest_path_labels = [G[path[i]][path[i + 1]]['label'] for i in range(len(path) - 1)]
    print("Shortest path from", start_node, "to", end_node, ":", path)
    print("Labels of the shortest path:", shortest_path_labels)
    combined_path = [str(path[i]) + f"--{shortest_path_labels[i]}-->" for i in range(len(path)-1)] + [str(path[-1])]
    print("Shortest path from", start_node, "to", end_node, ":", ''.join(combined_path))
# shortest_path = nx.shortest_path(G, source=start_node, target=end_node)  # 某一条最短路径
# print("Shortest path from", start_node, "to", end_node, ":", shortest_path)

plt.show()