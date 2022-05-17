
# ALUNOS:
# -Pedro Henrique Reis Rodrigues     -> Matrícula: 668443
# -Tárcila Fernanda Resende da Silva -> Matrícula: 680250
from sklearn import svm
import math
import os
from time import time
import tkinter
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import timeit
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


WIDTH, HEIGHT = 950, 500

test_images = []
test_labels = []
train_images = []
train_labels = []
x_test = []
GLCMaux = []
GLCM1 = []
GLCM2 = []
GLCM4 = []
GLCM8 = []
GLCM16 = []


# Criação da Matriz de Pixels através do TkInter
root = Tk()
root.title("Reconhecimento de Padrões por textura")

root.geometry("300x30+0+0")


def FeatureExtractor(dataset):
    # Função para a extração de dados de um vetor de imagens

    # Gera uma tabela de dados para cada imagem
    image_dataset = pd.DataFrame()
    global GLCMaux
    global GLCM1
    global GLCM2
    global GLCM4
    global GLCM8
    global GLCM16
    # Percorre por todas as imagens desse vetor
    for image in range(dataset.shape[0]):
        # Criação temporária de um data frame para guardar as informações de cada iteração
        df = pd.DataFrame()
        # Pega imagem do vetor tri-dimensional de imagens
        img = dataset[image, :, :]
        ##################################################
        # Geração de Matrizes Circulares C1, C2 ,C4 ,C8, C16

        # Declaração das matrizes de 4 dimensões, 32x32.

        GLCM1 = np.zeros(shape=(32, 32, 1, 1), dtype=int)
        GLCM2 = np.zeros(shape=(32, 32, 1, 1), dtype=int)
        GLCM4 = np.zeros(shape=(32, 32, 1, 1), dtype=int)
        GLCM8 = np.zeros(shape=(32, 32, 1, 1), dtype=int)
        GLCM16 = np.zeros(shape=(32, 32, 1, 1), dtype=int)

        for i in range(0, 8, 1):
            GLCM = greycomatrix(img, [1], [i*math.radians(360/8)], levels=32)
            for j in range(0, len(GLCM), 1):
                for k in range(0, len(GLCM[j]), 1):
                    GLCM1[j][k] += GLCM[j][k][0][0]

        for i in range(0, 16, 1):
            GLCM = greycomatrix(img, [2], [i*math.radians(360/16)], levels=32)
            for j in range(0, len(GLCM), 1):
                for k in range(0, len(GLCM[j]), 1):
                    GLCM2[j][k] += GLCM[j][k][0][0]

        for i in range(0, 24, 1):
            GLCM = greycomatrix(img, [4], [i*math.radians(360/24)], levels=32)
            for j in range(0, len(GLCM), 1):
                for k in range(0, len(GLCM[j]), 1):
                    GLCM4[j][k] += GLCM[j][k][0][0]

        for i in range(0, 48, 1):
            GLCM = greycomatrix(img, [8], [i*math.radians(360/48)], levels=32)
            for j in range(0, len(GLCM), 1):
                for k in range(0, len(GLCM[j]), 1):
                    GLCM8[j][k] += GLCM[j][k][0][0]

        for i in range(0, 96, 1):
            GLCM = greycomatrix(img, [16], [i*math.radians(360/96)], levels=32)
            for j in range(0, len(GLCM), 1):
                for k in range(0, len(GLCM[j]), 1):
                    GLCM16[j][k] += GLCM[j][k][0][0]

        # Extraindo propriedades de Haralick das matrizes de co-coocorrencia
        # Para a data frame df

        GLCM_Energy1 = greycoprops(GLCM1, 'energy')[0]
        df['Energy1'] = GLCM_Energy1
        GLCM_hom1 = greycoprops(GLCM1, 'homogeneity')[0]
        df['Homogen1'] = GLCM_hom1
        GLCM_entropy1 = shannon_entropy(GLCM1)
        df['Entropy1'] = GLCM_entropy1
        GLCM_corr1 = greycoprops(GLCM1, 'correlation')[0]
        df['Corr1'] = GLCM_corr1
        GLCM_contr1 = greycoprops(GLCM1, 'contrast')[0]
        df['Contrast1'] = GLCM_contr1
        GLCM_diss1 = greycoprops(GLCM1, 'dissimilarity')[0]
        df['Diss_sim1'] = GLCM_diss1

        GLCM_Energy2 = greycoprops(GLCM2, 'energy')[0]
        df['Energy2'] = GLCM_Energy2
        GLCM_hom2 = greycoprops(GLCM2, 'homogeneity')[0]
        df['Homogen2'] = GLCM_hom2
        GLCM_entropy2 = shannon_entropy(GLCM2)
        df['Entropy2'] = GLCM_entropy2
        GLCM_corr2 = greycoprops(GLCM2, 'correlation')[0]
        df['Corr2'] = GLCM_corr2
        GLCM_contr2 = greycoprops(GLCM2, 'contrast')[0]
        df['Contrast2'] = GLCM_contr2
        GLCM_diss2 = greycoprops(GLCM1, 'dissimilarity')[0]
        df['Diss_sim2'] = GLCM_diss2

        GLCM_Energy4 = greycoprops(GLCM4, 'energy')[0]
        df['Energy4'] = GLCM_Energy4
        GLCM_hom4 = greycoprops(GLCM4, 'homogeneity')[0]
        df['Homogen4'] = GLCM_hom4
        GLCM_entropy4 = shannon_entropy(GLCM4)
        df['Entropy4'] = GLCM_entropy4
        GLCM_corr4 = greycoprops(GLCM4, 'correlation')[0]
        df['Corr4'] = GLCM_corr4
        GLCM_contr4 = greycoprops(GLCM4, 'contrast')[0]
        df['Contrast4'] = GLCM_contr4
        GLCM_diss4 = greycoprops(GLCM4, 'dissimilarity')[0]
        df['Diss_sim4'] = GLCM_diss4

        GLCM_Energy8 = greycoprops(GLCM8, 'energy')[0]
        df['Energy8'] = GLCM_Energy8
        GLCM_hom8 = greycoprops(GLCM8, 'homogeneity')[0]
        df['Homogen8'] = GLCM_hom8
        GLCM_entropy8 = shannon_entropy(GLCM8)
        df['Entropy8'] = GLCM_entropy8
        GLCM_corr8 = greycoprops(GLCM8, 'correlation')[0]
        df['Corr8'] = GLCM_corr8
        GLCM_contr8 = greycoprops(GLCM8, 'contrast')[0]
        df['Contrast8'] = GLCM_contr8
        GLCM_diss8 = greycoprops(GLCM8, 'dissimilarity')[0]
        df['Diss_sim8'] = GLCM_diss8

        GLCM_Energy16 = greycoprops(GLCM16, 'energy')[0]
        df['Energy16'] = GLCM_Energy16
        GLCM_hom16 = greycoprops(GLCM16, 'homogeneity')[0]
        df['Homogen16'] = GLCM_hom16
        GLCM_entropy16 = shannon_entropy(GLCM16)
        df['Entropy16'] = GLCM_entropy16
        GLCM_corr16 = greycoprops(GLCM16, 'correlation')[0]
        df['Corr16'] = GLCM_corr16
        GLCM_contr16 = greycoprops(GLCM16, 'contrast')[0]
        df['Contrast16'] = GLCM_contr16
        GLCM_diss16 = greycoprops(GLCM16, 'dissimilarity')[0]
        df['Diss_sim16'] = GLCM_diss16

        # Insere na main dataset a auxiliar df da imagem "image"
        image_dataset = image_dataset.append(df)

    return image_dataset


def Training():
    start = timeit.default_timer()
    SIZE = 128
    global test_prediction
    global SVM_model
    global test_images
    global test_labels
    global train_images
    global train_labels
    global x_train
    global y_train
    global x_test
    global y_test
    cont = 0

    for directory_path in glob.glob("Treino/*"):
        label = directory_path.split("/")[-1]
        # print("label "+ label)
        # print(directory_path)
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            # print(img_path)
            # Lendo a imagem na escala de tons de cinza
            img = cv2.imread(img_path, 0)
            # Resize images por 128x128 (para via das dúvidas)
            img = cv2.resize(img, (SIZE, SIZE))
            tomMax = img.max()  # Pega o tom maximo da imagem para fazer a reamostragem em 32bits
            img32 = [[0 for x in range(128)] for y in range(128)]
            for i in range(0, 128, 1):
                for j in range(0, 128, 1):
                    # reamostragem em 32 bits pixel a pixel
                    img32[i][j] = np.uint8(round((img[i][j]/tomMax) * 31))
            train_images.append(img32)
            train_labels.append(label)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # Fazendo exatamente a mesma coisa para as imagens Teste

    for directory_path in glob.glob("Testes/*"):
        fruit_label = directory_path.split("/")[-1]
        cont = 0
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            # print(img_path)
            # Lendo a imagem na escala de tons de cinza
            img = cv2.imread(img_path, 0)
            # Resize images por 128x128 (para via das dúvidas)
            img = cv2.resize(img, (SIZE, SIZE))
            tomMax = img.max()  # Pega o tom maximo da imagem para fazer a reamostragem em 32bits
            cont += 1
            img32 = [[0 for x in range(128)] for y in range(128)]
            for i in range(0, 128, 1):
                for j in range(0, 128, 1):
                    # reamostragem em 32 bits pixel a pixel
                    img32[i][j] = np.uint8(round((img[i][j]/tomMax) * 31))
            test_images.append(img32)
            test_labels.append(fruit_label)
        # print(cont)
    # armazena a imagem e o label correspondente
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # Redefinição de variaveis a fim de melhor entendimento do código

    x_train = train_images
    y_train = train_labels
    x_test = test_images
    y_test = test_labels

    ####################################################################
    # Extrair as informações de haralick (atraves das matrizes de co-ocorrencia circulares)

    image_features = FeatureExtractor(x_train)
    X_for_ML = image_features

    # Definindo o classificador SVM

    SVM_model = svm.SVC(C=100, kernel='linear')

    # Modelando a estrutura de TREINO de dados para a SVM (Imagens,Labels)

    SVM_model.fit(X_for_ML, y_train)

    # Hora de testar nossa SVM
    # Extraindo as informações de haralick (atraves das matrizes de co-ocorrencia circulares) para as imagens TESTE agora

    test_features = FeatureExtractor(x_test)

    # Guarda os Predict -"Palpite"- da SVM na variavel

    test_prediction = SVM_model.predict(test_features)

    # # Set the parameters by cross-validation
    # tuned_parameters = [
    #     {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
    #     {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
    # ]

    # scores = ["precision", "recall"]

    # for score in scores:
    #     print("# Tuning hyper-parameters for %s" % score)
    #     print()

    #     clf = GridSearchCV(SVC(), tuned_parameters, scoring="%s_macro" % score)
    #     clf.fit(X_for_ML, y_train)

    #     print("Best parameters set found on development set:")
    #     print()
    #     print(clf.best_params_)
    #     print()
    #     print("Grid scores on development set:")
    #     print()
    #     means = clf.cv_results_["mean_test_score"]
    #     stds = clf.cv_results_["std_test_score"]
    #     for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
    #         print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    #     print()

    #     print("Detailed classification report:")
    #     print()
    #     print("The model is trained on the full development set.")
    #     print("The scores are computed on the full evaluation set.")
    #     print()
    #     y_true, y_pred = y_test, clf.predict(test_features)
    #     print(classification_report(y_true, y_pred))
    #     print()

    # # Note the problem is too easy: the hyperparameter plateau is too flat and the
    # # output model is the same for precision and recall with ties in quality.

    # Pop-up para mostrar  que o treino finalizou
    popSuccess = Toplevel(root)
    popSuccess.title("Treino da Rede Neural")
    popSuccess.geometry("300x100")
    popSuccess.config(bg="#C0C0C0")

    popSuccess_label = Label(popSuccess, text=f"O treino finalizou. ",
                             fg="black", font="Arial")
    popSuccess_label.pack(pady=10)

    stop = timeit.default_timer()
    time = stop - start
    time_label = Label(
        popSuccess, text=f'Time: {round(time,3)}s', fg="black", font=("Arial", 12))
    time_label.pack()
    time_label.place(x=0, y=0)


def RandomImageTesting():
    start = timeit.default_timer()
    # Função para checar e testar o palpite da SVM, utilizando aleatoriedade

    # Método simples para pegar qlqer numero dentro do intervalo do vetor de imagens testes
    n = random.randint(0, x_test.shape[0]-1)
    imgtest = x_test[n]

    # Descrevendo o vetor para virar imagem e ser exibido no canvas
    selectedImageArray = np.array(cv2.resize(imgtest, (400, 400)))
    selectedImage = ImageTk.PhotoImage(
        image=Image.fromarray(selectedImageArray))

    # Imagem selecionada
    pop = Toplevel(root)
    pop.title("Imagem Aleatória Selecionada")
    pop.geometry("450x450")
    pop.config(bg="#C0C0C0")
    canvas = Canvas(pop, width=450, height=450)
    canvas.create_image(25, 25, anchor=NW, image=selectedImage)
    canvas.pack()

    # Gera uma figura do estilo gráfico para mostrar os tons de cinza
    figure = Figure(figsize=(4, 4))
    ax = figure.add_subplot()
    ax.imshow(imgtest, cmap='gray')

    # Pop-up para mostrar o resultado
    resultPop = Toplevel(root)
    resultPop.title("Imagem Analisada e Processada")
    resultPop.geometry("500x520")
    resultPop.config(bg="#C0C0C0")

    titleLabel = Label(resultPop, text=f"Imagem reamostrada para 32 tons de cinza",
                       fg="black", font="Arial")
    titleLabel.pack()
    titleLabel.place(x=50, y=5)

    canvas2 = FigureCanvasTkAgg(figure, master=resultPop)
    canvas2.draw()
    canvas2.get_tk_widget().pack(pady=30)

    # Agora sim, depois de amostrar a imagem para o usuário, vamos fazer o palpite da SVM

    # Expande em 1 dimensão X para o input de imagens
    input_img = np.expand_dims(imgtest, axis=0)

    # Extrai os dados da imagem
    input_img_features = FeatureExtractor(input_img)

    # Predict da SVM

    img_prediction = SVM_model.predict(input_img_features)

    # Interface para mostrar o resultado

    label2 = Label(resultPop, text=f"A Rede Neural achou que a imagem era: BRADS{img_prediction[0]}\nE na verdade a imagem é: BRADS{test_labels[n]}",
                   fg="black", font="Arial")

    label2.place(x=50, y=435)
    # Cálculo da Acurácia / Especificidade
    accuracy = metrics.accuracy_score(test_labels, test_prediction)
    label4 = Label(resultPop, text=f"Acurácia = {accuracy}",
                   fg="black", font="Arial")
    label4.place(x=50, y=475)

    stop = timeit.default_timer()
    time = stop - start
    time_label = Label(
        resultPop, text=f'Time: {round(time,3)}s', fg="black", font=("Arial", 12))
    time_label.pack()
    time_label.place(x=0, y=0)


def TestSelectedImage():
    start = timeit.default_timer()
    # Função para checar e testar o palpite da SVM, utilizando a seleção do usuario

    global n
    global img2
    global filename2

    # Metódo exatamente o mesmo para selecionar a imagem no uploadImage()

    filename2 = os.path.abspath(tkf.askopenfilename(
        initialdir=os.getcwd(), title="Select your Image"))

    # Leitura da imagem em matriz (assim como foi feita na parte de Training())

    img2 = cv2.imread(filename2, 0)
    img2 = cv2.resize(img2, (128, 128))  # Resize por via das dúvidas

    # Interface para printar a imagem selecionada num pop-up

    selectedImageArray = np.array(cv2.resize(img2, (400, 400)))
    selectedImage = ImageTk.PhotoImage(
        image=Image.fromarray(selectedImageArray))

    pop = Toplevel(root)
    pop.title("Imagem Selecionada")
    pop.geometry("450x450")
    pop.config(bg="#C0C0C0")
    canvas = Canvas(pop, width=450, height=450)
    canvas.create_image(25, 25, anchor=NW, image=selectedImage)
    canvas.pack()

    # Mesmo procedimento do processo de treinamento

    # Aqui é necessário fazer a reamostragem por 32 tons de cinza novamente, pois como
    # Pegamos a imagem diretamente do diretório, ela ainda está como 255 tons de cinza

    img32 = [[0 for x in range(128)] for y in range(128)]
    for i in range(0, 128, 1):
        for j in range(0, 128, 1):
            img32[i][j] = np.uint8(round((img2[i][j]/255) * 31))

    img32 = np.array(img32)

    # Função básica para pegar qual a resposta do Teste através do nome do diretorio

    if(r"Testes\1" in filename2):
        n = 1
    elif(r"Testes\2" in filename2):
        n = 2
    elif(r"Testes\3" in filename2):
        n = 3
    elif(r"Testes\4" in filename2):
        n = 4

    # Gera uma figura do estilo gráfico para mostrar os tons de cinza

    imgtest = img32
    figure = Figure(figsize=(4, 4))
    ax = figure.add_subplot()
    ax.imshow(imgtest, cmap='gray')

    # Pop-up para mostrar o resultado
    resultPop = Toplevel(root)
    resultPop.title("Imagem Analisada e Processada")
    resultPop.geometry("500x520")
    resultPop.config(bg="#C0C0C0")

    titleLabel = Label(resultPop, text=f"Imagem reamostrada para 32 tons de cinza",
                       fg="black", font="Arial")
    titleLabel.pack()
    titleLabel.place(x=50, y=5)

    canvas2 = FigureCanvasTkAgg(figure, master=resultPop)
    canvas2.draw()
    canvas2.get_tk_widget().pack(pady=30)

    # Agora sim, depois de amostrar a imagem para o usuário, vamos fazer o palpite da SVM

    # Expande em 1 dimensão X para o input de imagens
    input_img = np.expand_dims(imgtest, axis=0)
    # Extrai os dados da imagem
    input_img_features = FeatureExtractor(input_img)
    # Predict da SVM
    img_prediction = SVM_model.predict(input_img_features)

    # Interface gráfica para mostrar o resultado

    label2 = Label(resultPop, text=f"A Rede Neural achou que a imagem era: BRADS{img_prediction[0]}\nE na verdade a imagem é: BRADS{n}",
                   fg="black", font="Arial")
    label2.place(x=50, y=435)
    # Cálculo da Acurácia / Especificidade
    accuracy = metrics.accuracy_score(test_labels, test_prediction)
    label4 = Label(resultPop, text=f"Acurácia = {accuracy}",
                   fg="black", font="Arial")
    label4.place(x=50, y=475)

    stop = timeit.default_timer()
    time = stop - start
    time_label = Label(
        resultPop, text=f'Time: {round(time,3)}s', fg="black", font=("Arial", 12))
    time_label.pack()
    time_label.place(x=0, y=0)


def printMatrixConfusion():
    start = timeit.default_timer()

    # Obtem matrix de confusão através da função pré definida do classificador SVM
    cm = confusion_matrix(test_labels, test_prediction)

    # Criação de figura para plotar usando matplot, + uso de uma biblioteca para fazer o heatmap do grafico

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

    # Cálculo da Acurácia
    cmaux = 0
    accuracy = metrics.accuracy_score(test_labels, test_prediction)
    # Cálculo da Especificidade
    for i in range(0, 4, 1):
        for j in range(0, 4, 1):
            if(i != j):
                cmaux += cm[j, i]

    specificity = round(1-(cmaux/300), 3)
    label2 = Label(pop, text=f"Acurácia = {accuracy}\nEspecificidade = {specificity}",
                   fg="black", font="Arial")
    label2.pack()
    label2.place(x=220, y=420)
    stop = timeit.default_timer()
    time = stop - start
    time_label = Label(
        pop, text=f'Time: {round(time,3)}s', fg="black", font=("Arial", 12))
    time_label.pack()
    time_label.place(x=0, y=0)


def FFTInfo():
    start = timeit.default_timer()

    # Obtenção da imagem através do diretorio pré-definido anteriormente pelo usuario "filename"

    img_reamostrada = io.imread(filename)

    # Reamostragem para 32 tons de cinza
    tomMax = img_reamostrada.max()
    for i in range(0, len(img_reamostrada), 1):
        for j in range(0, len(img_reamostrada[i]), 1):
            img_reamostrada[i][j] = (
                img_reamostrada[i][j]/tomMax)*31

    # Função geradora da FFT da biblioteca scipy, passando a imagem reamostrada como parametro

    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(img_reamostrada))

    # Passando o resultado para uma figura para ser exibida

    figure = Figure(figsize=(4, 4))
    ax = figure.add_subplot()
    ax.imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')

    # Pop-up para mostrar o resultado
    pop = Toplevel(root)
    pop.title("Transformada de Fourier")
    pop.geometry("500x500")
    pop.config(bg="#C0C0C0")

    init_figure = figure
    canvas2 = FigureCanvasTkAgg(init_figure, master=pop)
    canvas2.draw()
    canvas2.get_tk_widget().pack(pady=10)

    stop = timeit.default_timer()
    time = stop - start
    time_label = Label(
        pop, text=f'Time: {round(time,3)}s', fg="black", font=("Arial", 12))
    time_label.pack()
    time_label.place(x=0, y=0)


def dataInfo():

    start = timeit.default_timer()

    image = io.imread(filename)
    img32 = [[0 for x in range(128)] for y in range(128)]
    tomMax = image.max()
    for i in range(0, 128, 1):
        for j in range(0, 128, 1):
            img32[i][j] = np.uint8(round((image[i][j]/tomMax) * 31))
    # Gerando matriz de co-ocorrencia de 4 dimensões, no qual são 2 são para 1 distancia e 4 angulos
    GLCM1 = np.zeros(shape=(32, 32, 1, 1), dtype=int)
    GLCM2 = np.zeros(shape=(32, 32, 1, 1), dtype=int)
    GLCM4 = np.zeros(shape=(32, 32, 1, 1), dtype=int)
    GLCM8 = np.zeros(shape=(32, 32, 1, 1), dtype=int)
    GLCM16 = np.zeros(shape=(32, 32, 1, 1), dtype=int)

    for i in range(0, 8, 1):
        GLCM = greycomatrix(img32, [1], [i*math.radians(360/8)], levels=32)
        for j in range(0, len(GLCM), 1):
            for k in range(0, len(GLCM[j]), 1):
                GLCM1[j][k] += GLCM[j][k][0][0]

    for i in range(0, 16, 1):
        GLCM = greycomatrix(img32, [2], [i*math.radians(360/16)], levels=32)
        for j in range(0, len(GLCM), 1):
            for k in range(0, len(GLCM[j]), 1):
                GLCM2[j][k] += GLCM[j][k][0][0]

    for i in range(0, 24, 1):
        GLCM = greycomatrix(img32, [4], [i*math.radians(360/24)], levels=32)
        for j in range(0, len(GLCM), 1):
            for k in range(0, len(GLCM[j]), 1):
                GLCM4[j][k] += GLCM[j][k][0][0]

    for i in range(0, 48, 1):
        GLCM = greycomatrix(img32, [8], [i*math.radians(360/48)], levels=32)
        for j in range(0, len(GLCM), 1):
            for k in range(0, len(GLCM[j]), 1):
                GLCM8[j][k] += GLCM[j][k][0][0]

    for i in range(0, 96, 1):
        GLCM = greycomatrix(img32, [16], [i*math.radians(360/96)], levels=32)
        for j in range(0, len(GLCM), 1):
            for k in range(0, len(GLCM[j]), 1):
                GLCM16[j][k] += GLCM[j][k][0][0]

    GLCM_Energys = []
    GLCM_homs = []
    GLCM_entropys = []
    GLCM_correlations = []
    GLCM_contrasts = []
    GLCM_dissimilarity = []

    # GLCM propriedades
    GLCM_Energys.append(greycoprops(GLCM1, 'energy')[0][0])
    GLCM_homs.append(greycoprops(GLCM1, 'homogeneity')[0][0])
    GLCM_entropys.append(shannon_entropy(GLCM1))
    GLCM_correlations.append(greycoprops(GLCM1, 'correlation')[0][0])
    GLCM_contrasts.append(greycoprops(GLCM1, 'contrast')[0][0])
    GLCM_dissimilarity.append(greycoprops(GLCM1, 'dissimilarity')[0][0])

    GLCM_Energys.append(greycoprops(GLCM2, 'energy')[0][0])
    GLCM_homs.append(greycoprops(GLCM2, 'homogeneity')[0][0])
    GLCM_entropys.append(shannon_entropy(GLCM2))
    GLCM_correlations.append(greycoprops(GLCM2, 'correlation')[0][0])
    GLCM_contrasts.append(greycoprops(GLCM2, 'contrast')[0][0])
    GLCM_dissimilarity.append(greycoprops(GLCM2, 'dissimilarity')[0][0])

    GLCM_Energys.append(greycoprops(GLCM4, 'energy')[0][0])
    GLCM_homs.append(greycoprops(GLCM4, 'homogeneity')[0][0])
    GLCM_entropys.append(shannon_entropy(GLCM4))
    GLCM_correlations.append(greycoprops(GLCM4, 'correlation')[0][0])
    GLCM_contrasts.append(greycoprops(GLCM4, 'contrast')[0][0])
    GLCM_dissimilarity.append(greycoprops(GLCM4, 'dissimilarity')[0][0])

    GLCM_Energys.append(greycoprops(GLCM8, 'energy')[0][0])
    GLCM_homs.append(greycoprops(GLCM8, 'homogeneity')[0][0])
    GLCM_entropys.append(shannon_entropy(GLCM8))
    GLCM_correlations.append(greycoprops(GLCM8, 'correlation')[0][0])
    GLCM_contrasts.append(greycoprops(GLCM8, 'contrast')[0][0])
    GLCM_dissimilarity.append(greycoprops(GLCM8, 'dissimilarity')[0][0])

    GLCM_Energys.append(greycoprops(GLCM16, 'energy')[0][0])
    GLCM_homs.append(greycoprops(GLCM16, 'homogeneity')[0][0])
    GLCM_entropys.append(shannon_entropy(GLCM16))
    GLCM_correlations.append(greycoprops(GLCM16, 'correlation')[0][0])
    GLCM_contrasts.append(greycoprops(GLCM16, 'contrast')[0][0])
    GLCM_dissimilarity.append(greycoprops(GLCM16, 'dissimilarity')[0][0])

    # Pop-up para mostrar o resultado
    pop = Toplevel(root)
    pop.title("Descritores de Haralick da imagem 32 tons de cinza")
    pop.geometry("250x800")
    pop.config(bg="#C0C0C0")

    start = timeit.default_timer()

    # Your statements here

    for i in range(0, 5, 1):
        pop_label = Label(pop, text=f"Entropia C{(2**i)}: {round(GLCM_entropys[i],3)}\nEnergia C{2**i}: {round(GLCM_Energys[i],3)}\nHomogeneidade C{2**i}: {round(GLCM_homs[i],3)}\
        \nCorrelação C{2**i}: {round(GLCM_correlations[i],3)}\nContraste C{2**i}: {round(GLCM_contrasts[i],3)}\nDissimilaridade C{2**i}: {round(GLCM_dissimilarity[i],3)}",
                          fg="black", font="Arial")
        pop_label.pack()
        pop_label.place(x=10, y=50+(150 * (i)))

    stop = timeit.default_timer()
    time = stop - start
    time_label = Label(
        pop, text=f'Time: {round(time,3)}s', fg="black", font=("Arial", 12))
    time_label.pack()
    time_label.place(x=0, y=0)


def resampling(newMaxValue):

    # Pega a imagem que possui como destino o nome do diretorio filename
    # E aplica equalização definida previamente pelo usuario

    img_reamostrada = io.imread(filename)
    tomMax = img_reamostrada.max()  # Pega tom Max da imagem
    for i in range(0, len(img_reamostrada), 1):
        for j in range(0, len(img_reamostrada[i]), 1):
            img_reamostrada[i][j] = (
                img_reamostrada[i][j]/tomMax)*(newMaxValue - 1)  # Reamostragem por cada pixel

    # Visualização da Imagem Reamostrada

    figure = Figure(figsize=(4, 4))
    ax = figure.add_subplot()
    ax.imshow(img_reamostrada, cmap="gray")
    pop = Toplevel(root)
    pop.title(f"Imagem Reamostrada para {newMaxValue} tons de Cinza")
    pop.geometry("500x500")
    pop.config(bg="#C0C0C0")

    canvas2 = FigureCanvasTkAgg(figure, master=pop)
    canvas2.draw()
    canvas2.get_tk_widget().pack(pady=10)


def uploadImage():
    global img
    global filename

    # Armazena o diretorio da imagem desejada em filename
    # E redimensiona para 300x300 apenas para Visualização (não sera utilizada 300x300 nos calculos matrizes de co-ocorrencia)

    filename = os.path.abspath(tkf.askopenfilename(
        initialdir=os.getcwd(), title="Select your Image"))
    img = ImageTk.PhotoImage(Image.open(os.path.join(
        filename)).resize((300, 300), Image.ANTIALIAS))

    # Visualização da Imagem

    pop = Toplevel(root)
    pop.title("Imagem Original")
    pop.geometry("450x450")
    pop.config(bg="#C0C0C0")
    canvas = Canvas(pop, width=450, height=450)
    canvas.create_image(75, 75, anchor=NW, image=img)
    canvas.pack()


# ------------------------  Barra de menu ---------------------------
menubar = Menu(root)

# Opcoes para carregar a imagem
uploadMenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="Novo", menu=uploadMenu)
uploadMenu.add_command(label="Carregar imagem", command=uploadImage)
# Opcoes para aplicar descritores em imagem selecionada
imageMenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="Descrição", menu=imageMenu)
imageMenu.add_command(label="Obter Descritores Haralick", command=dataInfo)
imageMenu.add_command(label="Gerar Transformada de Fourier", command=FFTInfo)
resamplingMenu = Menu(imageMenu, tearoff=0)
resamplingMenu.add_command(label="32-bits", command=lambda: resampling(32))
resamplingMenu.add_command(label="16-bits", command=lambda: resampling(16))
resamplingMenu.add_command(label="8-bits", command=lambda: resampling(8))
resamplingMenu.add_command(label="4-bits", command=lambda: resampling(4))
resamplingMenu.add_command(label="2-bits", command=lambda: resampling(2))
imageMenu.add_cascade(label="Reamostrar imagem", menu=resamplingMenu)

# Opcoes para o reconhecimento de imagens
recognitionMenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="Reconhecimento", menu=recognitionMenu)
recognitionMenu.add_command(label="Treinar", command=Training)
testMenu = Menu(recognitionMenu, tearoff=0)
recognitionMenu.add_cascade(label="Teste", menu=testMenu)
testMenu.add_command(label="Com imagem aleatória", command=RandomImageTesting)
testMenu.add_command(label="Com imagem selecionada", command=TestSelectedImage)
recognitionMenu.add_command(
    label="Mostrar matriz de confusão", command=printMatrixConfusion)

root.config(menu=menubar)
root.mainloop()
