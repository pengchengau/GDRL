import numpy as np


def all_user_feature(U, T):
    Uf, Pu, Su, Ou, Vu, Lu = [], [], [], [], [], []

    for u in range(U):
        Uft, Put, Sut, Out, Vut, Lut = [], [], [], [], [], []

        for t in range(T):
            uf, pu, s_u, o_u, v_u, l_u = user_feature(t)
            Uft.append(uf)
            Put.append(pu)
            Sut.append(s_u)
            Out.append(o_u)
            Vut.append(v_u)
            Lut.append(l_u)

        Uf.append(Uft)
        Pu.append(Put)
        Su.append(Sut)
        Ou.append(Out)
        Vu.append(Vut)
        Lu.append(Lut)

    return np.array(Uf), np.array(Pu), np.array(Su), np.array(Ou), np.array(Vu), np.array(Lu)


def user_feature(t):
    if t % 10 == 0:
        uf = np.random.randint(1, 7)
    else:
        uf = np.random.randint(3, 7)

    pu_dBW = np.random.randint(10, 26)
    pu = 10 ** (pu_dBW / 10)

    if uf in {1, 2, 3}:
        s_u = np.random.randint(9000, 10001)
    else:
        s_u = np.random.randint(200, 301)

    o_u = np.random.randint(5000, 15001)

    if uf in {1, 2, 3}:
        v_u = np.random.randint(5000, 6001)
    else:
        v_u = np.random.randint(100, 151)

    if uf in {1, 2, 3}:
        l_u = np.random.randint(30, 100)
    else:
        l_u = np.random.randint(5, 11)

    return uf, pu, s_u, o_u, v_u, l_u