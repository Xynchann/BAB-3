import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Mengambil aksi yang tersedia berdasarkan matriks reward
def available_actions(state, R):
    current_state_row = R[state, :]
    av_act = np.where(current_state_row >= 0)[1] if len(R.shape) > 1 else np.where(current_state_row >= 0)[0]
    return av_act

# Memilih aksi secara acak dari aksi yang tersedia
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_actions_range, 1))
    return next_action

# Memperbarui nilai Q-Matrix
def update(current_state, action, gamma, R, Q):
    max_index = np.where(Q[action, :] == np.max(Q[action, :]))[0]
    if max_index.shape[0] > 0:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(action)
    max_value = Q[action, max_index]
    Q[current_state, action] = R[current_state, action] + gamma * max_value
    return Q

# Menampilkan grafik network
def showgraph(R):
    Graph = nx.from_numpy_array(R)
    nx.draw(Graph, with_labels=True)
    plt.show()

# Membuat matriks R awal
def createRmat(matrix_size, points_list):
    R = np.matrix(np.ones(matrix_size))
    R *= -1
    for point in points_list:
        R[point] = 0
    return R