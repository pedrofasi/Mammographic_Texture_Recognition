
# ALUNOS:
# -Pedro Henrique Reis Rodrigues     -> Matrícula: 668443
# -Tárcila Fernanda Resende da Silva -> Matrícula: 680250

import math
from operator import truediv
import os
from struct import pack
import tkinter
from cv2 import log
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
import random
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from skimage.filters import sobel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import cm

WIDTH, HEIGHT = 950, 500

test_images = []
test_labels = []
train_images = []
train_labels = []
x_test = []


# Criação da Matriz de Pixels através do TkInter
root = Tk()
root.title("Reconhecimento de Padrões por textura")
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
# root.geometry("1400x600+0+0")

# Definição de grid
canvasFrame = Frame(root)
canvasFrame.pack(side=tkinter.LEFT, fill=tkinter.X, expand=TRUE)

buttonsFrame = Frame(root, border=4)
buttonsFrame.pack(side=tkinter.RIGHT)

# Cria a janela, com tamanho de 1500x800p para seleção da imagem
canvas = Canvas(canvasFrame, width=WIDTH, height=HEIGHT,
                border=2, bg="#C0C0C0", cursor="dot")
canvas.pack(fill=tkinter.BOTH, expand=TRUE)


def BresenhamCircumference(xC, yC, r):
    x = 0
    y = r
    p = 3 + 2*r

    while(x < y):
        if(p < 0):
            p += 4 * x + 6
        else:
            p += 4*(x-y) + 10
            y -= 1
        x += 1


def entropy(GLCM):
    entropia = 0
    for i in range(GLCM):
        for j in range(GLCM[i]):
            entropia += GLCM[i][j] * math.log(GLCM[i][j], 2)

    return entropia


def FeatureExtractor(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):  # iterate through each file
        # print(image)

        # Temporary data frame to capture information for each loop.
        df = pd.DataFrame()
        # Reset dataframe to blank after each loop.

        img = dataset[image, :, :]
    ################################################################
    # START ADDING DATA TO THE DATAFRAME

        # Full image
        #GLCM = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
        GLCM1, GLCM2, GLCM4, GLCM8, GLCM16 = 0
        for i in range(0, 8, 1):
            GLCM1 += greycomatrix(img, [1], [i*(360/8)])

        for i in range(0, 16, 1):
            GLCM2 += greycomatrix(img, [1], [i*(360/16)])

        for i in range(0, 24, 1):
            GLCM4 += greycomatrix(img, [1], [i*(360/24)])

        for i in range(0, 48, 1):
            GLCM8 += greycomatrix(img, [1], [i*(360/48)])

        for i in range(0, 96, 1):
            GLCM16 += greycomatrix(img, [1], [i*(360/96)])

        GLCM_Energy1 = greycoprops(GLCM1, 'energy')[0]
        df['Energy1'] = GLCM_Energy1
        GLCM_hom1 = greycoprops(GLCM1, 'homogeneity')[0]
        df['Homogen1'] = GLCM_hom1
        GLCM_entropy1 = entropy(GLCM1)
        df['Entropy1'] = GLCM_entropy1

        GLCM_Energy2 = greycoprops(GLCM2, 'energy')[0]
        df['Energy2'] = GLCM_Energy2
        GLCM_hom2 = greycoprops(GLCM2, 'homogeneity')[0]
        df['Homogen2'] = GLCM_hom2
        GLCM_entropy2 = entropy(GLCM2)
        df['Entropy2'] = GLCM_entropy2

        GLCM_Energy4 = greycoprops(GLCM4, 'energy')[0]
        df['Energy4'] = GLCM_Energy4
        GLCM_hom4 = greycoprops(GLCM4, 'homogeneity')[0]
        df['Homogen4'] = GLCM_hom4
        GLCM_entropy4 = entropy(GLCM4)
        df['Entropy4'] = GLCM_entropy4

        GLCM_Energy8 = greycoprops(GLCM8, 'energy')[0]
        df['Energy8'] = GLCM_Energy8
        GLCM_hom8 = greycoprops(GLCM8, 'homogeneity')[0]
        df['Homogen8'] = GLCM_hom8
        GLCM_entropy1 = entropy(GLCM1)
        df['Entropy1'] = GLCM_entropy1

        GLCM_Energy16 = greycoprops(GLCM16, 'energy')[0]
        df['Energy16'] = GLCM_Energy16
        GLCM_hom16 = greycoprops(GLCM16, 'homogeneity')[0]
        df['Homogen16'] = GLCM_hom16
        GLCM_entropy1 = entropy(GLCM1)
        df['Entropy1'] = GLCM_entropy1

        # Add more filters as needed
        #entropy = shannon_entropy(img)
        #df['Entropy'] = entropy

        # Append features from current image to the dataset
        image_dataset = image_dataset.append(df)

    return image_dataset


def Training():
    SIZE = 128
    global imgtest
    global test_prediction
    global lgb_model
    global le
    global test_images
    global test_labels
    global train_images
    global train_labels
    global x_train
    global y_train
    global x_test
    global y_test

    # for directory_path in glob.glob("Treino/*"):
    for directory_path in glob.glob("Treino/*"):
        label = directory_path.split("\\")[-1]
        print(label)
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            print(img_path)
            # Lendo a imagem na escala de tons de cinza
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (SIZE, SIZE))  # Resize images
            train_images.append(img)
            train_labels.append(label)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # Do exactly the same for test/validation images
    # test

    # for directory_path in glob.glob("cell_images/test/*"):
    for directory_path in glob.glob("Testes/*"):
        fruit_label = directory_path.split("\\")[-1]
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (SIZE, SIZE))
            test_images.append(img)
            test_labels.append(fruit_label)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # Encode labels from text (folder names) to integers.

    le = preprocessing.LabelEncoder()
    le.fit(test_labels)
    test_labels_encoded = le.transform(test_labels)
    le.fit(train_labels)
    train_labels_encoded = le.transform(train_labels)

    # Split data into test and train datasets (already split but assigning to meaningful convention)
    # If you only have one dataset then split here
    x_train = train_images
    y_train = train_labels_encoded
    x_test = test_images
    y_test = test_labels_encoded

    # Normalize pixel values to between 0 and 1
    #x_train, x_test = x_train / 255.0, x_test / 255.0

    ###################################################################
    # FEATURE EXTRACTOR function
    # input shape is (n, x, y, c) - number of images, x, y, and channels

    ####################################################################
    # Extract features from training images
    image_features = FeatureExtractor(x_train)
    X_for_ML = image_features
    # Reshape to a vector for Random Forest / SVM training
    #n_features = image_features.shape[1]
    #image_features = np.expand_dims(image_features, axis=0)
    # X_for_ML = np.reshape(image_features, (x_train.shape[0], -1))  #Reshape to #images, features

    # Define the classifier
    # from sklearn.ensemble import RandomForestClassifier
    # RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)

    # Can also use SVM but RF is faster and may be more accurate.
    #from sklearn import svm
    # SVM_model = svm.SVC(decision_function_shape='ovo')  #For multiclass classification
    #SVM_model.fit(X_for_ML, y_train)

    # Fit the model on training data
    # RF_model.fit(X_for_ML, y_train) #For sklearn no one hot encoding

    # Class names for LGBM start at 0 so reassigning labels from 1,2,3,4 to 0,1,2,3
    d_train = lgb.Dataset(X_for_ML, label=y_train)

    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
    lgbm_params = {'learning_rate': 0.05, 'boosting_type': 'dart',
                   'objective': 'multiclass',
                   'metric': 'multi_logloss',
                   'num_leaves': 100,
                   'max_depth': 10,
                   'num_class': 4}  # no.of unique values in the target class not inclusive of the end value

    # 50 iterations. Increase iterations for small learning rates
    lgb_model = lgb.train(lgbm_params, d_train, 100)

    # Predict on Test data
    # Extract features from test data and reshape, just like training data
    test_features = FeatureExtractor(x_test)
    test_features = np.expand_dims(test_features, axis=0)
    test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))

    # Predict on test
    test_prediction = lgb_model.predict(test_for_RF)
    test_prediction = np.argmax(test_prediction, axis=1)
    # Inverse le transform to get original label back.
    test_prediction = le.inverse_transform(test_prediction)

    # Pop-up para mostrar  que o treino finalizou
    popSuccess = Toplevel(root)
    popSuccess.title("Treino da Rede Neural")
    popSuccess.geometry("300x100")
    popSuccess.config(bg="#C0C0C0")

    popSuccess_label = Label(popSuccess, text=f"O treino finalizou. ",
                             fg="black", font="Arial")
    popSuccess_label.pack(pady=10)


def RandomImageTesting():
    # Check results on a few random images
    # Select the index of image to be loaded for testing
    global auximg2
    n = random.randint(0, x_test.shape[0]-1)
    imgtest = x_test[n]

    print("\n\n imagem teste \n", imgtest)

    image = cv2.cvtColor(imgtest, cv2.COLOR_BGR2RGB)
    image = ImageTk.PhotoImage(image=Image.fromarray(image))
    auximg2 = image
    canvas.delete('all')
    labelImgTitle = Label(root, text=f"Imagem utilizada no teste:",
                          fg="black", font="Arial")
    labelImgTitle.pack()
    labelImgTitle.place(x=50, y=35)
    canvas.create_image(50, 60, anchor=NW, image=auximg2)

    figure = Figure(figsize=(4, 4))
    ax = figure.add_subplot()
    ax.imshow(imgtest)

    # Pop-up para mostrar a imagem processada
    pop = Toplevel(root)
    pop.title("Imagem Analisada e Processada")
    pop.geometry("500x500")
    pop.config(bg="#C0C0C0")

    canvas2 = FigureCanvasTkAgg(figure, master=pop)
    canvas2.draw()
    canvas2.get_tk_widget().pack(pady=10)

    # Extract features and reshape to right dimensions
    # Expand dims so the input is (num images, x, y, c)
    input_img = np.expand_dims(imgtest, axis=0)
    input_img_features = FeatureExtractor(input_img)
    input_img_features = np.expand_dims(input_img_features, axis=0)
    input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))
    # Predict
    img_prediction = lgb_model.predict(input_img_for_RF)
    img_prediction = np.argmax(img_prediction, axis=1)
    # Reverse the label encoder to original name
    img_prediction = le.inverse_transform([img_prediction])
    label2 = Label(root, text=f"A Rede Neural achou que a imagem era:{img_prediction}\nE na verdade a imagem é: {test_labels[n]}",
                   fg="black", font="Arial")
    label2.pack()
    label2.place(x=50, y=250)

    aux = metrics.accuracy_score(test_labels, test_prediction)
    label4 = Label(root, text=f"Accuracy = {aux}",
                   fg="black", font="Arial")
    label4.pack()
    label4.place(x=50, y=300)


def TestSelectedImage():
    # Check results on a few random images
    # Select the index of image to be loaded for testing
    global n
    global img2
    global filename2
    global auximg
    filename2 = os.path.abspath(tkf.askopenfilename(
        initialdir=r'C:\\', title="Select your Image"))
    img2 = ImageTk.PhotoImage(Image.open(os.path.join(filename2)))
    auximg = img2

    canvas.delete('all')
    labelImgTitle = Label(root, text=f"Imagem utilizada no teste:",
                          fg="black", font="Arial")
    labelImgTitle.pack()
    labelImgTitle.place(x=50, y=35)

    canvas.create_image(50, 60, anchor=NW, image=auximg)

    if(r"Testes/1" in filename2):
        n = 1
    elif(r"Testes/2" in filename2):
        n = 2

    elif(r"Testes/3" in filename2):
        n = 3

    elif(r"Testes/4" in filename2):
        n = 4

    img2 = cv2.imread(filename2, 0)
    imgtest = img2
    figure = Figure(figsize=(4, 4))
    ax = figure.add_subplot()
    ax.imshow(imgtest)

    # Pop-up para mostrar o resultado
    pop = Toplevel(root)
    pop.title("Imagem Analisada e Processada")
    pop.geometry("500x500")
    pop.config(bg="#C0C0C0")

    canvas2 = FigureCanvasTkAgg(figure, master=pop)
    canvas2.draw()
    canvas2.get_tk_widget().pack(pady=10)

    # Extract features and reshape to right dimensions
    # Expand dims so the input is (num images, x, y, c)
    input_img = np.expand_dims(imgtest, axis=0)
    input_img_features = FeatureExtractor(input_img)
    input_img_features = np.expand_dims(input_img_features, axis=0)
    input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))
    # Predict
    img_prediction = lgb_model.predict(input_img_for_RF)
    img_prediction = np.argmax(img_prediction, axis=1)
    # Reverse the label encoder to original name
    img_prediction = le.inverse_transform([img_prediction])
    label2 = Label(root, text=f"A Rede Neural achou que a imagem era:{img_prediction}\nE na verdade a imagem é: {n}",
                   fg="black", font="Arial")
    label2.pack()
    label2.place(x=50, y=250)

    # label3 = Label(root, text=f"Imagem Analisada e Processada:",
    #                fg="black", font=("Arial", 20), )
    # label3.pack()
    # label3.place(x=500, y=50)

    aux = metrics.accuracy_score(test_labels, test_prediction)
    label4 = Label(root, text=f"Accuracy = {aux}",
                   fg="black", font="Arial")
    label4.pack()
    label4.place(x=50, y=300)


def printMatrixConfusion():
    cm = confusion_matrix(test_labels, test_prediction)

    figure = Figure(figsize=(4, 4))
    ax = figure.subplots()
    sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)

    # Pop-up para mostrar o resultado
    pop = Toplevel(root)
    pop.title("Matriz de confusão")
    pop.geometry("500x500")
    pop.config(bg="#C0C0C0")

    init_figure = figure
    canvas2 = FigureCanvasTkAgg(init_figure, master=pop)
    canvas2.draw()
    canvas2.get_tk_widget().pack(pady=10)
    # canvas2.get_tk_widget().place(x=50, y=80)

    aux = metrics.accuracy_score(test_labels, test_prediction)
    label2 = Label(pop, text=f"Accuracy = {aux}",
                   fg="black", font="Arial")
    label2.pack()
    label2.place(x=220, y=420)


def dataInfo():
    global label
    image = io.imread(filename)

    matrix_coocurrence = greycomatrix(
        image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])

    # printando a matriz de coocorrencia
    # for i in range(0, len(matrix_coocurrence), 1):
    #     for j in range(0, len(matrix_coocurrence[i]), 1):
    #         if(matrix_coocurrence[i][j].any() != 0):
    #             print(matrix_coocurrence[i][j][0])

    # GLCM propriedades
    homogeneidade = greycoprops(matrix_coocurrence, 'homogeneity')[0, 0]
    energia = greycoprops(matrix_coocurrence, 'energy')[0, 0]
    entropia = shannon_entropy(image)

    # Pop-up para mostrar o resultado
    pop = Toplevel(root)
    pop.title("Descritores de Haralick da imagem")
    pop.geometry("400x100")
    pop.config(bg="#C0C0C0")

    pop_label = Label(pop, text=f"Entropia: {entropia}\nEnergia: {energia}\nHomogeneidade: {homogeneidade}",
                      fg="black", font="Arial")
    pop_label.pack(pady=10)


def uploadImage():
    global img
    global filename
    filename = os.path.abspath(tkf.askopenfilename(
        initialdir=r'C:\\', title="Select your Image"))
    img = ImageTk.PhotoImage(Image.open(os.path.join(filename)))
    canvas.delete("all")
    canvas.create_image(120, 120, anchor=NW, image=img)

# Definindo botões de ações -----------------------------------------------------------------------------------------------


butUpload = Button(buttonsFrame, text="Selecionar uma imagem", bg="#696969",
                   fg="WHITE", activebackground="#4F4F4F", width=40, command=uploadImage)
butUpload.grid(row=1, column=1, padx=20, pady=20)

butGetDATA = Button(buttonsFrame, text="Descrever imagem", bg="#696969",
                    fg="WHITE", activebackground="#4F4F4F", width=40, command=dataInfo)
butGetDATA.grid(row=2, column=1, padx=20, pady=20)


butTrain = Button(buttonsFrame, text="Treinar rede neural", bg="#696969",
                  fg="WHITE", activebackground="#4F4F4F", width=40, command=Training)
butTrain.grid(row=3, column=1, padx=20, pady=20)

butTest = Button(buttonsFrame, text="Testar a rede neural com uma imagem aleatória",
                 bg="#696969", fg="WHITE", activebackground="#4F4F4F", width=40, command=RandomImageTesting)
butTest.grid(row=4, column=1, padx=20, pady=20)

butTestImage = Button(buttonsFrame, text="Testar a rede neural com uma imagem selecionada",
                      bg="#696969", fg="WHITE", activebackground="#4F4F4F", width=40, command=TestSelectedImage)
butTestImage.grid(row=5, column=1, padx=20, pady=20)

butConfusionMatrix = Button(buttonsFrame, text="Printar Matriz de Confusão", bg="#696969",
                            fg="WHITE", activebackground="#4F4F4F", width=40, command=printMatrixConfusion)
butConfusionMatrix.grid(row=6, column=1, padx=20, pady=10)


root.mainloop()
