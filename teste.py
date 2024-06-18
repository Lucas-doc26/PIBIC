import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from modelos import carrega_modelo
from preprocessamento import *
from visualizacao import plot_confusion_matrix

# Desativar TF_ENABLE_ONEDNN_OPTS para evitar problemas com o TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Carregar modelo previamente treinado
modelo_PUC_Congelado = carrega_modelo('Modelos_keras/PUC_Congelado_mobilenetv3.keras', 'weights_finais/PUC_Congelado_mobilenetv3.weights.h5')

# Carregar dados para teste (pode ser alterado para 'teste' ou 'val')
csv, df_g, df = preprocessamento_dataframe_teste("Datasets_csv/df_PUC.csv")

# Obter caminhos das imagens e rótulos reais
caminhos_imagens = df['caminho_imagem'].tolist()
rotulos_reais = df['classe'].values  
print(rotulos_reais)

# Mapear rótulos para binário
rotulos_binarios = mapear_rotulos_binarios(rotulos_reais)

# Carregar e pré-processar imagens
imagens = carregar_e_preprocessar_imagens(caminhos_imagens)
predicoes = modelo_PUC_Congelado.predict(imagens).argmax(axis=1)

# Definir labels para a matriz de confusão
labels = ['Empty', 'Occupied']

# Calcular a matriz de confusão
matriz_confusao = confusion_matrix(rotulos_binarios, predicoes)

# Exibir a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao, display_labels=labels)
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, values_format='.0f')
plt.title('Matriz de Confusão')
plt.xlabel('Predições')
plt.ylabel('Rótulos Reais')
plt.grid(False)
plt.show()
