import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from typing import List

def preprocessamento(caminho:str) -> List[tuple]:
    """
    Ao passar um dataFrame .csv, ele irá retornar uma tupla com 3 dataFrames gerados: treino, teste e validação
    """
    
    dataframe = pd.read_csv(caminho)

    treino, teste = train_test_split(dataframe, test_size=0.2, random_state=42)
    treino, validacao = train_test_split(treino, test_size=0.25, random_state=42)

    img_width, img_height = 256, 256
    batch_size = 32

    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    treino_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    validacao_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    teste_dategen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    treino_gerador = treino_datagen.flow_from_dataframe(
            dataframe=treino,
            x_col='caminho_imagem',
            y_col='classe', 
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )


    validacao_gerador = validacao_datagen.flow_from_dataframe(
            dataframe=validacao,
            x_col='caminho_imagem',
            y_col='classe',
            target_sizgeradorModelosConvMobileNetV3e=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )

    teste_gerador = teste_dategen.flow_from_dataframe(
            dataframe=teste,
            x_col='caminho_imagem',
            y_col='classe',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
    return treino_gerador, validacao_gerador, teste_gerador, treino, validacao, teste

def preprocessamento_dataframe_teste(caminho:str):
    """
    Retorna o dataFrame lido pelo csv e o dataFrame gerado pelo ImageDataGenerator
    """
    
    img_width, img_height = 256, 256
    batch_size = 32
    dataframe = pd.read_csv(caminho)
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    dataframe_gerador = datagen.flow_from_dataframe(
            dataframe=dataframe,
            x_col='caminho_imagem',
            y_col='classe',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary'
        )
    
    return dataframe, dataframe_gerador, dataframe

def carregar_e_preprocessar_imagens(caminhos_imagens, target_size=(256, 256)):
    imagens = []
    for caminho in caminhos_imagens:
        img = load_img(caminho, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        imagens.append(img_array)
    return np.vstack(imagens)

def mapear_rotulos_binarios(rotulos):
    return np.array([1 if r == 'Occupied' else 0 for r in rotulos])