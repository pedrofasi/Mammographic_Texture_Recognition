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


WIDTH, HEIGHT = 1500, 800

# ALUNO: Pedro Henrique Reis Rodrigues
# MATRÍCULA: 668443

# Criação da Matriz de Pixels através do TkInter
root = Tk()

# Cria a janela, com tamanho de 1500x800p para seleção da imagem
canvas = Canvas(root, width=WIDTH, height=HEIGHT)
canvas.pack()

test_images = []
test_labels = []
train_images = []
train_labels = []


def Training():
    for directory_path in glob.glob("Treino/*"):
        label = directory_path.split("\\")[-1]
        print(label)
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            print(img_path)
            img = cv2.imread(img_path, 0)  # le as cores da imagem
            train_images.append(img)
            train_labels.append(label)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)


def Testing():
    for directory_path in glob.glob("Testes/*"):
        fruit_label = directory_path.split("\\")[-1]

        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            print(img_path)
            img = cv2.imread(img_path, 0)  # le as cores da imagem
            train_images.append(img)
            train_labels.append(fruit_label)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)


def feature_extractor(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):
        df = pd.DataFrame()
        imagem = dataset[image, :, :]

        GLCM = greycomatrix(imagem, [1], [0])
        GLCM_Energy = greycoprops(GLCM, 'energy')[0]
        df['Energy'] = GLCM_Energy
        GLCM_entropy = shannon_entropy(GLCM)
        df['Entropy'] = GLCM_entropy
        GLCM_hom = greycoprops(GLCM, 'homogeneity')[0]
        df['Homogen'] = GLCM_hom

        GLCM2 = greycomatrix(img, [3], [0])
        GLCM_Energy2 = greycoprops(GLCM2, 'energy')[0]
        df['Energy2'] = GLCM_Energy2
        GLCM_entropy2 = shannon_entropy(GLCM2)
        df['Entropy2'] = GLCM_entropy2
        GLCM_hom2 = greycoprops(GLCM2, 'homogeneity')[0]
        df['Homogen2'] = GLCM_hom2

        GLCM3 = greycomatrix(img, [5], [0])
        GLCM_Energy3 = greycoprops(GLCM3, 'energy')[0]
        df['Energy3'] = GLCM_Energy3
        GLCM_entropy3 = shannon_entropy(GLCM3)
        df['Entropy3'] = GLCM_entropy3
        GLCM_hom3 = greycoprops(GLCM3, 'homogeneity')[0]
        df['Homogen3'] = GLCM_hom3

        GLCM4 = greycomatrix(img, [0], [np.pi/4])
        GLCM_Energy4 = greycoprops(GLCM4, 'energy')[0]
        df['Energy4'] = GLCM_Energy4
        GLCM_entropy4 = shannon_entropy(GLCM4)
        df['Entropy4'] = GLCM_entropy4
        GLCM_hom4 = greycoprops(GLCM4, 'homogeneity')[0]
        df['Homogen4'] = GLCM_hom4

        GLCM5 = greycomatrix(img, [0], [np.pi/2])
        GLCM_Energy5 = greycoprops(GLCM5, 'energy')[0]
        df['Energy5'] = GLCM_Energy5
        GLCM_entropy5 = shannon_entropy(GLCM5)
        df['Entropy5'] = GLCM_entropy5
        GLCM_hom5 = greycoprops(GLCM5, 'homogeneity')[0]
        df['Homogen5'] = GLCM_hom5

        image_dataset = image_dataset.append(df)
    return image_dataset


def Learning():
    le = preprocessing.LabelEncoder()
    le.fit(test_labels)
    test_labels_encoded = le.transform(test_labels)
    le.fit(train_labels)
    train_labels_encoded = le.transform(train_labels)
    x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded
    image_features = feature_extractor(x_train)
    X_for_ML = image_features


def dataInfo():
    global lista
    global total_rows
    global total_columns
    image = io.imread('p_d_left_cc(12).png')
    matrix_coocurrence = greycomatrix(
        image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])

    # GLCM propriedades
    homogeneidade = greycoprops(matrix_coocurrence, 'homogeneity')[0, 0]
    energia = greycoprops(matrix_coocurrence, 'energy')[0, 0]
    entropia = shannon_entropy(image)

    data = {'Entropia': [entropia],
            'Energia': [energia],
            'Homogeneidade': [homogeneidade]
            }
    df = DataFrame(data, columns=['Entropia', 'Energia', 'Homogeneidade'])
    print(df)


def uploadImage():
    global img
    filename = os.path.abspath(tkf.askopenfilename(
        initialdir=r'C:\Users\Fasi\Desktop\PI Trabalho', title="Select your Image"))
    img = ImageTk.PhotoImage(Image.open(os.path.join(filename)))
    canvas.create_image(60, 60, anchor=NW, image=img)


butUpload = Button(root, text="Upload Image", activebackground="black",
                   command=uploadImage, width=10)
butUpload.pack()
butUpload.place(x=10, y=10)

butGetDATA = Button(root, text="Get Img Data", activebackground="black",
                    command=dataInfo, width=15)
butGetDATA.pack()
butGetDATA.place(x=100, y=10)


root.mainloop()
