import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import keras
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint
from segmentandoDatasets import segmentadando_datasets, csv_para_dicionario
from visualizacao import *
from preprocessamento import preprocessamento, preprocessamento_dataframe_teste
from modelos import carrega_modelo

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

modelo_PUC_Congelado = carrega_modelo('Modelos_keras/PUC_Congelado_mobilenetv3.keras','weights_finais/PUC_Congelado_mobilenetv3.weights.h5')

_, _, dataset_PUC = preprocessamento("Datasets_csv/df_PUC.csv")

x_val, y_val = next(dataset_PUC)

print(x_val, y_val)

predicoes = modelo_PUC_Congelado.predict(dataset_PUC)
#print(predicoes)
predicoes = np.argmax(predicoes, axis=1)
#print(predicoes)

conf_matrix = confusion_matrix(dataset_PUC.labels, predicoes)

"""print(conf_matrix)"""

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['empty', 'occupied'], 
            yticklabels=['empty', 'occupied'])
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()


