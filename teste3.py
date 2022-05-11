import math
import os
import tkinter
from pandas import DataFrame
import tkinter.filedialog as tkf
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage import io
from skimage.measure import shannon_entropy
import cv2
import glob
from sklearn import preprocessing
import pandas as pd

matrix = np.zeros(shape=(32, 32), dtype=int)

image = io.imread('p_d_left_cc(12).png')
img32 = [[0 for x in range(128)] for y in range(128)]
for i in range(0, 128, 1):
    for j in range(0, 128, 1):
        img32[i][j] = np.uint8((image[i][j]/255) * 31)

print(image)
print(len(img32), len(img32[0]))
matrix_coocurrence = greycomatrix(
    img32, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32)

# printando a matriz de coocorrencia
print(len(matrix_coocurrence))
print(len(matrix_coocurrence[0]))
print(len(matrix_coocurrence[0][0][0]))
for i in range(0, len(matrix_coocurrence), 1):
    for j in range(0, len(matrix_coocurrence[0]), 1):
        if(matrix_coocurrence[i][j].any() != 0):
            for k in range(0, len(matrix_coocurrence[i][j][0])):
                matrix[i][j] += matrix_coocurrence[i][j][0][k]
print(len(matrix))
print(len(matrix[0]))
for i in range(0, len(matrix), 1):
    print(matrix[i])
