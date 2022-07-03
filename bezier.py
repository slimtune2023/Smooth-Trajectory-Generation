from matplotlib import pyplot as plt

def gen_point(A, k):
    p = [0, 0]
    
    p[0] += A[0][0] * ((1 - k) ** 3)
    p[1] += A[1][0] * ((1 - k) ** 3)
    
    p[0] += A[0][1] * (3 * k * (1 - k) ** 2)
    p[1] += A[1][1] * (3 * k * (1 - k) ** 2)
    
    p[0] += A[0][2] * (3 * (k ** 2) * (1 - k))
    p[1] += A[1][2] * (3 * (k ** 2) * (1 - k))
    
    p[0] += A[0][3] * (k ** 3)
    p[1] += A[1][3] * (k ** 3)
    
    return p

def gen_look_up(num, A):
    t = []
    look_up = []
    prev = [A[0][0], A[1][0]]

    for i in range(num + 1):
        ti = i / num

        cur = gen_point(A, ti)

        dist = ((cur[0] - prev[0]) ** 2 + (cur[1] - prev[1]) ** 2) ** 0.5

        if (i == 0):
            look_up.append(dist)
        else:
            look_up.append(dist + look_up[i - 1])

        t.append(ti)
        prev = cur
    
    return t, look_up

def bin_search(l, val):
    low = 0
    high = len(l) - 1
    
    while (high > low + 1):
        mid = (low + high) // 2
        
        if (l[mid] <= val):
            low = mid
        else:
            high = mid
    
    return low

def gen_path(n, t, look_up):
    x = []
    y = []

    k = []
    dist = []

    total_dist = look_up[-1]
    prev_index = 0

    for i in range(n + 1):
        dist_i = total_dist * i / n
        index = prev_index + bin_search(look_up[prev_index:], dist_i)

        k_i = ((dist_i - look_up[index]) / (look_up[index + 1] - look_up[index])) * (t[index + 1] - t[index]) + t[index]
        k.append(k_i)
        dist.append(dist_i)

        x_i, y_i = gen_point(P, k_i)
        x.append(x_i)
        y.append(y_i)

        prev_index = index
    
    return x, y, k, dist

def gen_limits(P):
    lims = [P[0][0], P[0][0], P[1][0], P[1][0]] # minX, maxX, minY, maxY

    for i in range(len(P[0])):
        if (P[0][i] < lims[0]):
            lims[0] = P[0][i]
        elif (P[0][i] > lims[1]):
            lims[1] = P[0][i]

        if (P[1][i] < lims[2]):
            lims[2] = P[1][i]
        elif (P[1][i] > lims[3]):
            lims[3] = P[1][i]
    
    return lims

def plot_total_dist(look_up_k, look_up_dist, path_k, path_dist):
    plt.xlim(0, 1)
    plt.ylim(0, look_up_dist[-1])
    plt.grid()
    plt.plot(look_up_k, look_up_dist, marker=".", markersize=4, markeredgecolor="red")
    plt.plot(path_k, path_dist, marker="X", markersize=8, markeredgecolor="blue")
    plt.show()
  
def plot_path(P, path_x, path_y):
    plt.xlim(limits[0] - 1, limits[1] + 1)
    plt.ylim(limits[2] - 1, limits[3] + 1)
    plt.grid()
    plt.plot(P[0], P[1], marker="o", markersize=10, color = "orange", markeredgecolor="orange", markerfacecolor="orange", animated = True)
    plt.plot(path_x, path_y, marker=".", markersize=5, markeredgecolor="blue", animated = True)
    plt.show()

def print_delta_dist(path_x, path_y):
    prev = [0, 0]
    
    for i in range(len(path_x)):
        if (i != 0):
            print(((path_x[i] - prev[0]) ** 2 + (path_y[i] - prev[1]) ** 2) ** 0.5)

        prev = [path_x[i], path_y[i]]

P = [[-1, 2, 3, -0.5],
     [4, 9, 1, 7]]

limits = gen_limits(P)
look_up_k, look_up_dist = gen_look_up(10000, P)
path_x, path_y, path_k, path_dist = gen_path(50, look_up_k, look_up_dist)

plot_total_dist(look_up_k, look_up_dist, path_k, path_dist)

plot_path(P, path_x, path_y)
