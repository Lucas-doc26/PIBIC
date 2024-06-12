import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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
            class_mode='binary'
        )


    validacao_gerador = validacao_datagen.flow_from_dataframe(
            dataframe=validacao,
            x_col='caminho_imagem',
            y_col='classe',
            target_sizgeradorModelosConvMobileNetV3e=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary'
        )

    teste_gerador = teste_dategen.flow_from_dataframe(
            dataframe=teste,
            x_col='caminho_imagem',
            y_col='classe',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary'
        )
    return treino_gerador, validacao_gerador, teste_gerador

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
    
    return dataframe, dataframe_gerador
