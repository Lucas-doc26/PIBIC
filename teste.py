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
from modelos import *
from TransferLearning import teste_modelos

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


PUCPR_dic = csv_para_dicionario('Datasets_csv\df_PUC.csv')
teste_modelos(PUCPR_dic, 0)

ufpr = csv_para_dicionario('Datasets_csv/df_UFPR04.csv')
_, dataset_1 = preprocessamento_dataframe_teste(ufpr)

modelo = carrega_modelo('Modelos_keras/PUC_Congelado_mobilenetv3.keras','weights_finais/PUC_Congelado_mobilenetv3.weights.h5')
predicoes = modelo.predict(dataset_1)
predicoes = np.argmax(predicoes, axis=1)
print(predicoes)


