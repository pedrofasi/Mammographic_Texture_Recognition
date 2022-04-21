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
import random
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import lightgbm as lgb
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
from skimage.filters import sobel
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import cm

WIDTH, HEIGHT = 1500, 800

test_images = []
test_labels = []
train_images = []
train_labels = []
x_test = []

# ALUNOS:
# -Pedro Henrique Reis Rodrigues
# -Tárcila Fernanda Resende da Silva
# MATRÍCULA:
# -668443
# -680250


# Criação da Matriz de Pixels através do TkInter
root = Tk()

# Cria a janela, com tamanho de 1500x800p para seleção da imagem
canvas = Canvas(root, width=WIDTH, height=HEIGHT)
canvas.pack()


def feature_extractor(dataset):
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
        GLCM = greycomatrix(img, [1], [0])
        GLCM_Energy = greycoprops(GLCM, 'energy')[0]
        df['Energy'] = GLCM_Energy
        GLCM_corr = greycoprops(GLCM, 'correlation')[0]
        df['Corr'] = GLCM_corr
        GLCM_diss = greycoprops(GLCM, 'dissimilarity')[0]
        df['Diss_sim'] = GLCM_diss
        GLCM_hom = greycoprops(GLCM, 'homogeneity')[0]
        df['Homogen'] = GLCM_hom
        GLCM_contr = greycoprops(GLCM, 'contrast')[0]
        df['Contrast'] = GLCM_contr

        GLCM2 = greycomatrix(img, [3], [0])
        GLCM_Energy2 = greycoprops(GLCM2, 'energy')[0]
        df['Energy2'] = GLCM_Energy2
        GLCM_corr2 = greycoprops(GLCM2, 'correlation')[0]
        df['Corr2'] = GLCM_corr2
        GLCM_diss2 = greycoprops(GLCM2, 'dissimilarity')[0]
        df['Diss_sim2'] = GLCM_diss2
        GLCM_hom2 = greycoprops(GLCM2, 'homogeneity')[0]
        df['Homogen2'] = GLCM_hom2
        GLCM_contr2 = greycoprops(GLCM2, 'contrast')[0]
        df['Contrast2'] = GLCM_contr2

        GLCM3 = greycomatrix(img, [5], [0])
        GLCM_Energy3 = greycoprops(GLCM3, 'energy')[0]
        df['Energy3'] = GLCM_Energy3
        GLCM_corr3 = greycoprops(GLCM3, 'correlation')[0]
        df['Corr3'] = GLCM_corr3
        GLCM_diss3 = greycoprops(GLCM3, 'dissimilarity')[0]
        df['Diss_sim3'] = GLCM_diss3
        GLCM_hom3 = greycoprops(GLCM3, 'homogeneity')[0]
        df['Homogen3'] = GLCM_hom3
        GLCM_contr3 = greycoprops(GLCM3, 'contrast')[0]
        df['Contrast3'] = GLCM_contr3

        GLCM4 = greycomatrix(img, [0], [np.pi/4])
        GLCM_Energy4 = greycoprops(GLCM4, 'energy')[0]
        df['Energy4'] = GLCM_Energy4
        GLCM_corr4 = greycoprops(GLCM4, 'correlation')[0]
        df['Corr4'] = GLCM_corr4
        GLCM_diss4 = greycoprops(GLCM4, 'dissimilarity')[0]
        df['Diss_sim4'] = GLCM_diss4
        GLCM_hom4 = greycoprops(GLCM4, 'homogeneity')[0]
        df['Homogen4'] = GLCM_hom4
        GLCM_contr4 = greycoprops(GLCM4, 'contrast')[0]
        df['Contrast4'] = GLCM_contr4

        GLCM5 = greycomatrix(img, [0], [np.pi/2])
        GLCM_Energy5 = greycoprops(GLCM5, 'energy')[0]
        df['Energy5'] = GLCM_Energy5
        GLCM_corr5 = greycoprops(GLCM5, 'correlation')[0]
        df['Corr5'] = GLCM_corr5
        GLCM_diss5 = greycoprops(GLCM5, 'dissimilarity')[0]
        df['Diss_sim5'] = GLCM_diss5
        GLCM_hom5 = greycoprops(GLCM5, 'homogeneity')[0]
        df['Homogen5'] = GLCM_hom5
        GLCM_contr5 = greycoprops(GLCM5, 'contrast')[0]
        df['Contrast5'] = GLCM_contr5

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
    # for directory_path in glob.glob("cell_images/train/*"):
    for directory_path in glob.glob("Treino/*"):
        label = directory_path.split("\\")[-1]
        print(label)
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            print(img_path)
            img = cv2.imread(img_path, 0)  # Reading color images
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
    image_features = feature_extractor(x_train)
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
    test_features = feature_extractor(x_test)
    test_features = np.expand_dims(test_features, axis=0)
    test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))

    # Predict on test
    test_prediction = lgb_model.predict(test_for_RF)
    test_prediction = np.argmax(test_prediction, axis=1)
    # Inverse le transform to get original label back.
    test_prediction = le.inverse_transform(test_prediction)


def RandomImageTesting():
    # Check results on a few random images
    # Select the index of image to be loaded for testing
    global auximg2
    n = random.randint(0, x_test.shape[0]-1)
    imgtest = x_test[n]

    image = cv2.cvtColor(imgtest, cv2.COLOR_BGR2RGB)
    image = ImageTk.PhotoImage(image=Image.fromarray(image))
    auximg2 = image
    canvas.create_image(600, 60, anchor=NW, image=auximg2)

    figure = Figure(figsize=(4, 4))
    ax = figure.add_subplot()
    ax.imshow(imgtest)

    canvas2 = FigureCanvasTkAgg(figure, master=root)
    canvas2.draw()
    canvas2.get_tk_widget().pack()
    canvas2.get_tk_widget().place(x=1000, y=100)

    # Extract features and reshape to right dimensions
    # Expand dims so the input is (num images, x, y, c)
    input_img = np.expand_dims(imgtest, axis=0)
    input_img_features = feature_extractor(input_img)
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
    label2.place(x=500, y=220)

    label3 = Label(root, text=f"Imagem Analisada e Processada:",
                   fg="black", font=("Arial", 20), )
    label3.pack()
    label3.place(x=1020, y=50)

    aux = metrics.accuracy_score(test_labels, test_prediction)
    label4 = Label(root, text=f"Accuracy = {aux}",
                   fg="black", font=("Arial", 20))
    label4.pack()
    label4.place(x=1080, y=700)


def printMatrixConfusion():
    cm = confusion_matrix(test_labels, test_prediction)

    figure = Figure(figsize=(4, 4))
    ax = figure.subplots()
    sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)

    init_figure = figure
    canvas = FigureCanvasTkAgg(init_figure, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()
    canvas.get_tk_widget().place(x=1000, y=100)

    label3 = Label(root, text=f"\tConfusion Matrix:\t",
                   fg="black", font=("Arial", 20), )
    label3.pack()
    label3.place(x=1080, y=50)

    aux = metrics.accuracy_score(test_labels, test_prediction)
    label2 = Label(root, text=f"Accuracy = {aux}",
                   fg="black", font=("Arial", 20))
    label2.pack()
    label2.place(x=1080, y=700)


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
    canvas.create_image(600, 60, anchor=NW, image=auximg)

    if(r"Testes\1" in filename2):
        n = 1
    elif(r"Testes\2" in filename2):
        n = 2
    elif(r"Testes\3" in filename2):
        n = 3
    elif(r"Testes\4" in filename2):
        n = 4

    img2 = cv2.imread(filename2, 0)
    imgtest = img2
    figure = Figure(figsize=(4, 4))
    ax = figure.add_subplot()
    ax.imshow(imgtest)

    canvas2 = FigureCanvasTkAgg(figure, master=root)
    canvas2.draw()
    canvas2.get_tk_widget().pack()
    canvas2.get_tk_widget().place(x=1000, y=100)

    # Extract features and reshape to right dimensions
    # Expand dims so the input is (num images, x, y, c)
    input_img = np.expand_dims(imgtest, axis=0)
    input_img_features = feature_extractor(input_img)
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
    label2.place(x=500, y=220)

    label3 = Label(root, text=f"Imagem Analisada e Processada:",
                   fg="black", font=("Arial", 20), )
    label3.pack()
    label3.place(x=1020, y=50)

    aux = metrics.accuracy_score(test_labels, test_prediction)
    label4 = Label(root, text=f"Accuracy = {aux}",
                   fg="black", font=("Arial", 20))
    label4.pack()
    label4.place(x=1080, y=700)


def dataInfo():
    global label
    image = io.imread(filename)
    matrix_coocurrence = greycomatrix(
        image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])

    # printando a matriz de coocorrencia
    # for i in range(0, len(matrix_coocurrence), 1):
    # for j in range(0, len(matrix_coocurrence[i]), 1):
    # if(matrix_coocurrence[i][j].any() != 0):
    # print(matrix_coocurrence[i][j][0])

    # GLCM propriedades
    homogeneidade = greycoprops(matrix_coocurrence, 'homogeneity')[0, 0]
    energia = greycoprops(matrix_coocurrence, 'energy')[0, 0]
    entropia = shannon_entropy(image)

    label = Label(root, text=f"Entropia: {entropia}\nEnergia: {energia}\nHomogeneidade: {homogeneidade}",
                  fg="black", font="Arial")
    label.pack()
    label.place(x=0, y=200)


def uploadImage():
    global img
    global filename
    filename = os.path.abspath(tkf.askopenfilename(
        initialdir=r'C:\Users\Fasi\Desktop\PI Trabalho', title="Select your Image"))
    img = ImageTk.PhotoImage(Image.open(os.path.join(filename)))
    canvas.create_image(60, 60, anchor=NW, image=img)


butUpload = Button(root, text="Upload Image", activebackground="black",
                   command=uploadImage, width=11)
butUpload.pack()
butUpload.place(x=10, y=10)

butGetDATA = Button(root, text="Get Image Data", activebackground="black",
                    command=dataInfo, width=15)
butGetDATA.pack()
butGetDATA.place(x=110, y=10)

butTrain = Button(root, text="Train Neural Network", activebackground="black",
                  command=Training, width=19)
butTrain.pack()
butTrain.place(x=240, y=10)

butTest = Button(root, text="Test Neural Network with a Random Image", activebackground="black",
                 command=RandomImageTesting, width=35)
butTest.pack()
butTest.place(x=395, y=10)

butTestImage = Button(root, text="Test Neural Network with a Selected Image", activebackground="black",
                      command=TestSelectedImage, width=35)
butTestImage.pack()
butTestImage.place(x=665, y=10)

butConfusionMatrix = Button(root, text="Print Confusion Matrix", activebackground="black",
                            command=printMatrixConfusion, width=20)
butConfusionMatrix.pack()
butConfusionMatrix.place(x=935, y=10)

root.mainloop()
