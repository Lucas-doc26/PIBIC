import os
from typing import Dict
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


#os.environ["CUDNN_PATH"] = "/home/lucas/Documents/.venv/lib/python3.11/site-packages/nvidia/cudnn"
#os.environ["LD_LIBRARY_PATH"] = "$CUDNN_PATH:$LD_LIBRARY_PATH"
#os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda/"


def teste_modelos(caminho:str, num:int):
    treino, validacao, teste = preprocessamento(caminho)
    model = cria_modelo_mobileNetV3(num)
    model.summary(show_trainable=True)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    checkpoint_path = 'weights_parciais/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.weights.h5'
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, 
                                  save_weights_only=True, 
                                  monitor='val_accuracy', 
                                  mode='max', 
                                  save_best_only=True, 
                                  verbose=1)

    history = model.fit(treino,
                        epochs=10,
                        callbacks=[cp_callback],
                        validation_data=validacao)

    dataset_paths = {
        'PUC': 'Datasets_csv/df_PUC.csv',
        'UFPR04': 'Datasets_csv/df_UFPR04.csv',
        'UFPR05': 'Datasets_csv/df_UFPR05.csv'
    }

    nome_Modelo = next((key for key in dataset_paths if key in caminho), 'UFPR05')
    model_save_path = f"Modelos_keras/{nome_Modelo}_{'Congelado' if num == 0 else 'Descongelado'}_mobilenetv3.keras"
    weights_save_path = f"weights_finais/{nome_Modelo}_{'Congelado' if num == 0 else 'Descongelado'}_mobilenetv3.weights.h5"

    model.save(model_save_path)
    model.save_weights(weights_save_path)

    filename = f"Resultados/teste_{nome_Modelo}_{'Congelado' if num == 0 else 'Descongelado'}"
    with open(filename, 'w') as f:
        _, dataset_1 = preprocessamento_dataframe_teste(dataset_paths.get(nome_Modelo))
        _, dataset_2 = preprocessamento_dataframe_teste('Datasets_csv/df_UFPR05.csv' if nome_Modelo != 'UFPR05' else 'Datasets_csv/df_UFPR04.csv')

        modelo = keras.models.load_model(model_save_path)  # Carregar o modelo original

        modelo.load_weights(weights_save_path)

        test_results = {
            f"{nome_Modelo} X {nome_Modelo}": modelo.evaluate(teste)[1],
            f"{nome_Modelo} X {'UFPR04' if nome_Modelo == 'PUC' else 'PUC'}": modelo.evaluate(dataset_1)[1],
            f"{nome_Modelo} X {'UFPR05' if nome_Modelo != 'UFPR05' else 'UFPR04'}": modelo.evaluate(dataset_2)[1]
        }

        for modelos, accuracy in test_results.items():
            print(f"{modelos} - precisão de {accuracy:.3f} ", file=f)

def predict_e_matriz():

    PUC = preprocessamento_dataframe_teste('Datasets_csv/df_PUC.csv')
    UFPR04 = preprocessamento_dataframe_teste('Datasets_csv/df_UFPR04.csv')
    UFPR05 = preprocessamento_dataframe_teste('Datasets_csv/df_UFPR05.csv')

    modelos = {
        'PUC_Congelado': 'Modelos_keras\\PUC_Congelado_mobilenetv3.keras',
        'PUC_Descongelado' : 'Modelos_keras\\PUC_Descongelado_mobilenetv3.keras', 
        'UFPR04_Congelado': 'Modelos_keras\\UFPR04_Congelado_mobilenetv3.keras',
        'UFPR04_Descongelado': 'Modelos_keras\\UFPR04_Descongelado_mobilenetv3.keras',
        'UFPR05_Congelado': 'Modelos_keras\\PUC_Congelado_mobilenetv3.keras',
        'UFPR05_Descongelado': 'Modelos_keras\\PUC_Descongelado_mobilenetv3.keras',
    }

    pesos = {
        'PUC_Congelado': 'weights_finais\\PUC_Congelado_mobilenetv3.h5',
        'PUC_Descongelado' : 'weights_finais\\PUC_Descongelado_mobilenetv3.h5', 
        'UFPR04_Congelado': 'weights_finais\\UFPR04_Congelado_mobilenetv3.h5',
        'UFPR04_Descongelado': 'weights_finais\\UFPR04_Congelado_mobilenetv3.h5',
        'UFPR05_Congelado': 'weights_finais\\UFPR05_Congelado_mobilenetv3.h5',
        'UFPR05_Descongelado': 'weights_finais\\UFPR05_Congelado_mobilenetv3.h5',
    }

    for modelo_nome, modelo_caminho in modelos.items():
        pesos_caminho = pesos[modelo_nome]
        modelo = carrega_modelo(modelo_caminho, pesos_caminho)
        print(modelo.summary())

        if 'PUC' in modelo_nome:
            y_true = PUC[1].classes
            y_pred = modelo.predict(PUC)[1].argmax(axis=1)
        elif 'UFPR04' in modelo_nome:
            y_true = UFPR04[1].classes
            y_pred = modelo.predict(UFPR04)[1].argmax(axis=1)
        elif 'UFPR05' in modelo_nome:
            y_true = UFPR05[1].classes
            y_pred = modelo.predict(UFPR05)[1].argmax(axis=1)

        labels = ['Empty', 'Occupied']  # Certifique-se de ajustar os rótulos conforme necessário
        plot_confusion_matrix(y_true, y_pred, labels)


#Exemplo de criação dos modelos:
print("Modelos sem o Fine Tuning:\n")
teste_modelos('Datasets_csv/df_UFPR04.csv', 0)
teste_modelos('Datasets_csv/df_UFPR05.csv', 0)

print("\nModelos com o Fine Tuning:\n")
teste_modelos('Datasets_csv/df_PUC.csv', 2)
teste_modelos('Datasets_csv/df_UFPR04.csv', 2)
teste_modelos('Datasets_csv/df_UFPR05.csv', 2)


predict_e_matriz()