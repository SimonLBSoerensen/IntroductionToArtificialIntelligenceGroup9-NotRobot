import matplotlib.pyplot as plt
import numpy as np

map = np.array([
    ["#"]*8,
    ["#", " ", " ", " ", " ", " ", " ", "#"],
    ["#", " ", " ", " ", " ", " ", " ", "#"],
    ["#", " ", " ", "C", " ", " ", "D", "#"],
    ["#", " ", " ", " ", " ", "C", " ", "#"],
    ["#", " ", "D", " ", " ", " ", " ", "#"],
    ["#", " ", " ", " ", " ", " ", " ", "#"],
    ["#"]*8,
])

tail_color = {
    "#": [0.0, 0.0, 0.0],
    "P": [0.0, 1.0, 0.0],
    "C": [1.0, 0.0, 0.0],
    "D": [0.0, 0.0, 1.0],
    " ": [1.0, 1.0, 1.0],
}


def draw_map(map, tail_color):
    n, m = map.shape
    map_img = np.full((n, m, 3), tail_color[" "])
    for tail in tail_color:
        x, y = np.where(map == tail)
        map_img[x,y] = tail_color[tail]
    return map_img


def show_map(map_img):
    plt.figure()
    plt.imshow(map_img)
    plt.show()

map_img = draw_map(map, tail_color)

show_map(map_img)