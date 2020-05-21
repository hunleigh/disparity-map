import numpy as np
from PIL import Image
import random
import math


"""
---------------------------------------------
functions
---------------------------------------------
"""


# returns the difference between two pixels
def elemCost(lmap, rmap, i, j, y):
    rpix = rmap[j, y]
    lpix = lmap[i, y]
    return ((rpix - lpix) ** 2) / 4


# create a black and white image filled with noise
def createStereo(image):
    greyscalearr = [0, 255]
    imgpix = image.load()
    for x in range(image.width):
        for y in range(image.height):
            imgpix[x, y] = random.choice(greyscalearr)


# appends a line in the disparity matrix
def appendLine(image, width, height, arr):
    pmap = image.load()
    for i in range(len(arr)):
        if arr[i] == -1:
            pmap[i, height] = 0
        else:
            pmap[i, height] = int(arr[i] * 1.5 + 128)


# returns the index of the min elem
def imin(elem1, elem2, elem3):
    cmin = min(elem1, elem2, elem3)
    if cmin == elem1:
        return 1
    elif cmin == elem2:
        return 2
    else:
        return 3


# back track to find optimal pixel values
def bpass(C_d, C_v, C_h, w):
    # start from bottom corner

    i = j = w

    global d_arr
    d_arr[:] = 0

    # determine initial conditions based on if pixels match

    init_cmin = imin(C_d[i, j], C_v[i, j], C_h[i, j])
    if init_cmin == 1:  # if i matches with j
        d_arr[i - 1] = abs(i - j)
        c1 = 0
        c2 = 1
        c3 = 1
        i1 = 1
        j1 = 1
    elif init_cmin == 2:  # if i doesn't match
        d_arr[i - 1] = -1
        c1 = 1
        c2 = 0
        c3 = 1
        i1 = 1
        j1 = 0
    else:  # if j doesn't match
        d_arr[j - 1] = -1
        c1 = 1
        c2 = 1
        c3 = 0
        i1 = 0
        j1 = 1

    while i > 0 and j > 0:
        cmin = imin(C_d[i - i1, j - j1] + c1, C_v[i - i1, j - j1] + c2,
                    C_h[i - i1, j - j1] + c3)  # minimum cost
        if cmin == 1:
            d_arr[i - i1 - 1] = abs(i - j)  # i - i1 matches with j - j1
            c1 = 0
            c2 = 1
            c3 = 1
            i -= i1
            j -= j1
            i1 = 1
            j1 = 1
        elif cmin == 2:
            d_arr[i - i1 - 1] = -1  # i - i1 doesn't match
            c1 = 1
            c2 = 0
            c3 = 1
            i -= i1
            j -= j1
            i1 = 1
            j1 = 0
        else:
            d_arr[j - j1 - 1] = -1  # j - j1 doesn't match
            c1 = 1
            c2 = 1
            c3 = 0
            i -= i1
            j -= j1
            i1 = 0
            j1 = 1

    return d_arr


# forward pass with maximum horizontal likelihood
def fpass(lmap, rmap, x, y):
    global disparityMap
    global C, C_d, C_v, C_h

    # clear the cost matrices for each line
    C[1:, 1:] = 0
    C_d[:, :] = 0
    C_v[:, :] = 0
    C_h[:, :] = 0

    maxx = np.infty

    for i in range(1, x + 1):
        for j in range(1, y + 1):
            d = C[i - 1, j - 1] + elemCost(lmap, rmap, i - 1, j - 1, y)
            h = C[i, j - 1] + occlusion
            v = C[i - 1, j] + occlusion
            C[i, j] = cmin = min(d, h, v)
            if cmin == d:
                C_d[i, j] = imin(C_d[i - 1, j - 1], C_h[i - 1, j - 1] + 1,
                                 C_v[i - 1, j - 1] + 1)
            else:
                C_d[i, j] = maxx
            if cmin == h:
                C_h[i, j] = imin(C_d[i - 1, j] + 1, C_h[i - 1, j],
                                 C_v[i - 1, j] + 1)
            else:
                C_h[i, j] = maxx
            if cmin == v:
                C_v[i, j] = imin(C_d[i, j - 1] + 1, C_h[i, j - 1] + 1,
                                 C_v[i, j - 1])
            else:
                C_v[i, j] = maxx

    appendLine(disparityMap, x, y, bpass(C_d, C_v, C_h, x))


"""
---------------------------------------------
variables
---------------------------------------------
"""
path = "/Users/user/Desktop/algorithms/CW2/"

leftImg = Image.open(path + 'v1.png').convert("L")
rightImg = Image.open(path + 'v2.png').convert("L")
leftpixelmap = leftImg.load()
rightpixelmap = rightImg.load()
w, h = leftImg.size
d_arr = np.zeros(w, dtype=int)
C = np.zeros((w + 1, w + 1), dtype=np.int)  # cost matrices
C_d = np.zeros((w + 1, w + 1), dtype=np.int)
C_v = np.zeros((w + 1, w + 1), dtype=np.int)
C_h = np.zeros((w + 1, w + 1), dtype=np.int)

disparityMap = Image.new('L', (w, h), 255)

# cost of occlusion
occlusion = 4

# initialise cost matrix
for i in range(w + 1):
    C[i, 0] = i * occlusion
    C[0, i] = i * occlusion


for h_0 in range(int(h / 2)):
    fpass(leftpixelmap, rightpixelmap, w, h_0)
    fpass(leftpixelmap, rightpixelmap, w, h_0 + int(h / 2))

disparityMap.save(path + "disparity map.png")
disparityMap.show()

leftImg.save(path + "left greyscale.png")
rightImg.save(path + "right greyscale.png")
leftImg.show()
