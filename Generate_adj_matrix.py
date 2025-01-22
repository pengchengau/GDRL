import itertools
import numpy as np


def GenerateAdjacency(U, L, N) :
    Adj_Matrix = np.zeros((U + L + N, U + L + N))
    for u in range(U):
        # Each user is connected to two LEO: user 1 to LEO 1 and LEO 2
        Adj_Matrix[u, U + u] = 1
        Adj_Matrix[u, U + u + 1] = 1
    for u in range(U, U + U + 1):
        # Each access LEO can connect with other non-access LEOs with the connection number 0-4
        numbers_LEO = np.arange(U + U + 1, U + L)
        connection_L = np.arange(1, 4)
        number_connection_L = np.random.choice(connection_L, 1, replace=False)[0]
        index = np.random.choice(numbers_LEO, number_connection_L, replace=False)
        for i in range(number_connection_L):
            Adj_Matrix[u, index[i]] = 1
    for u in range(U):
        # Each user is connected to two HAPS: user 1 to HAPS 1 and HAPS 2
        Adj_Matrix[u, U + L + u] = 1
        Adj_Matrix[u, U + L + u + 1] = 1
    for u in range(U + L, U + U + L + 1):
        # Each access HAPS can connect with other non-access HAPS with the connection number 0-4
        numbers_HAPS = np.arange(U + U + L + 1, U + L + N)
        connection_N = np.arange(1, 4)
        number_connection_N = np.random.choice(connection_N, 1, replace=False)[0]
        index_n = np.random.choice(numbers_HAPS, number_connection_N, replace=False)
        for i in range(number_connection_N):
            Adj_Matrix[u, index_n[i]] = 1
    Adj_Matrix = np.triu(Adj_Matrix, 1) + np.triu(Adj_Matrix, 1).T  # make matrix symmetric

    # Calculate the possible offloading decision for each user u (The LEO/HAPS connected with user directly or indirectly)
    user_lists = {}  # 创建一个空字典，用于保存每个用户的列表
    for u in range(U):
        # 创建一个空的列表来保存user u的二进制字符串
        binary_u = []
        # 将直接连接的LEO从十进制转为2进制（LEO编码：0-15）
        binary_u_1 = bin(u)[2:].zfill(7)
        binary_u_2 = bin(u + 1)[2:].zfill(7)
        binary_u.extend([binary_u_1, binary_u_2])
        # 将间接链接的LEO转变为2进制，需要注意的是左侧开始第一位0为LEO标识，第二位0为直接链接，为1为间接链接的LEO
        indices = np.where(Adj_Matrix[u + U, U:U + L] == 1)
        for i in range(len(indices[0])):
            binary_u_3 = bin(indices[0][i])[3:].zfill(4)
            binary_u_3 = "010" + binary_u_3
            binary_u.extend([binary_u_3])
        indices = np.where(Adj_Matrix[u + U + 1, U:U + L] == 1)
        for i in range(len(indices[0])):
            binary_u_4 = bin(indices[0][i])[3:].zfill(4)
            binary_u_4 = "010" + binary_u_4
            binary_u.extend([binary_u_4])
        # 将直接连接的HAPS从十进制转为2进制（HAPS编码：0-15）
        binary_u_1 = "100" + bin(u)[3:].zfill(4)
        binary_u_2 = "100" + bin(u + 1)[3:].zfill(4)
        binary_u.extend([binary_u_1, binary_u_2])
        # 间接链接的HAPS
        indices = np.where(Adj_Matrix[u + U + L, U + L:U + L + N] == 1)
        for i in range(len(indices[0])):
            binary_u_3 = bin(indices[0][i])[3:].zfill(4)
            binary_u_3 = "110" + binary_u_3
            binary_u.extend([binary_u_3])
        indices = np.where(Adj_Matrix[u + U + L + 1, U + L:U + L + N] == 1)
        for i in range(len(indices[0])):
            binary_u_4 = bin(indices[0][i])[3:].zfill(4)
            binary_u_4 = "110" + binary_u_4
            binary_u.extend([binary_u_4])
        binary_u.extend(["0010000"])
        # 将列表添加到字典中，以用户编号u作为键
        user_lists[u] = binary_u
    # Make sure that no same action in the list
    for u in range(U):
        user_lists[u] = list(set(user_lists[u]))
    # Convert the binary to 10-based
    user_lists_new = {}
    for u in range(U):
        convert_user_lists = [int(item, 2) for item in user_lists[u]]
        user_lists_new[u] = convert_user_lists

    # Calculate the total number of action for all users
    action_space_len = 0
    for u in range(U):
        action_space_len += len(user_lists_new[u])

    return action_space_len, Adj_Matrix, user_lists_new
