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
from visualizacao import plot_confusion_matrix
from preprocessamento import preprocessamento, preprocessamento_dataframe_teste
from modelos import carrega_modelo, cria_modelo_mobileNetV3


os.environ["CUDNN_PATH"] = "/home/lucas/Documents/.venv/lib/python3.11/site-packages/nvidia/cudnn"
os.environ["LD_LIBRARY_PATH"] = "$CUDNN_PATH:$LD_LIBRARY_PATH"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda/"


def teste_modelos(caminho:str, num:int):
    """
    Compila e treina o modelo e faz evaluate com todos os datasets
    """

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
#Exemplo de criação dos modelos:
"""
print("Modelos sem o Fine Tuning:\n")
teste_modelos('Datasets_csv/df_PUC.csv', 0)
teste_modelos('Datasets_csv/df_UFPR04.csv', 0)
teste_modelos('Datasets_csv/df_UFPR05.csv', 0)

print("\nModelos com o Fine Tuning:\n")
teste_modelos('Datasets_csv/df_PUC.csv', 2)
teste_modelos('Datasets_csv/df_UFPR04.csv', 2)
teste_modelos('Datasets_csv/df_UFPR05.csv', 2)
"""


def predict_e_matriz_de_confusao():
    PUC_csv, PUC = preprocessamento_dataframe_teste('Datasets_csv/df_PUC.csv')
    UFPR04_csv, UFPR04 = preprocessamento_dataframe_teste('Datasets_csv/df_UFPR04.csv')
    UFPR05_csv, UFPR05 = preprocessamento_dataframe_teste('Datasets_csv/df_UFPR05.csv')

    modelos = {
        'PUC_Congelado': r'/home/lucas/PIBIC/Modelos_keras/PUC_Congelado_mobilenetv3.keras',
        'PUC_Descongelado': r'/home/lucas/PIBIC/Modelos_keras/PUC_Descongelado_mobilenetv3.keras', 
        'UFPR04_Congelado': r'/home/lucas/PIBIC/Modelos_keras/UFPR04_Congelado_mobilenetv3.keras',
        'UFPR04_Descongelado': r'/home/lucas/PIBIC/Modelos_keras/UFPR04_Descongelado_mobilenetv3.keras',
        'UFPR05_Congelado': r'/home/lucas/PIBIC/Modelos_keras/UFPR05_Congelado_mobilenetv3.keras',
        'UFPR05_Descongelado': r'/home/lucas/PIBIC/Modelos_keras/UFPR05_Descongelado_mobilenetv3.keras',
    }

    pesos = {
        'PUC_Congelado': r'/home/lucas/PIBIC/weights_finais/PUC_Congelado_mobilenetv3.weights.h5',
        'PUC_Descongelado': r'/home/lucas/PIBIC/weights_finais/PUC_Descongelado_mobilenetv3.weights.h5', 
        'UFPR04_Congelado': r'/home/lucas/PIBIC/weights_finais/UFPR04_Congelado_mobilenetv3.weights.h5',
        'UFPR04_Descongelado': r'/home/lucas/PIBIC/weights_finais/UFPR04_Descongelado_mobilenetv3.weights.h5',
        'UFPR05_Congelado': r'/home/lucas/PIBIC/weights_finais/UFPR05_Congelado_mobilenetv3.weights.h5',
        'UFPR05_Descongelado': r'/home/lucas/PIBIC/weights_finais/UFPR05_Descongelado_mobilenetv3.weights.h5',
    }

    modelos_carregados = {}

    for modelo_nome, modelo_caminho in modelos.items():
        pesos_caminho = pesos[modelo_nome]

        modelo = keras.models.load_model(modelo_caminho)
        modelo.load_weights(pesos_caminho)

        modelos_carregados[modelo_nome] = modelo 

    print(modelos_carregados)

    for modelo_nome, modelo in modelos_carregados.items():
        if 'PUC' in modelo_nome: 
            y_verdadeiro1 = PUC.classes
            y_predicao1 = modelo.predict(PUC).argmax(axis=1)

            y_verdadeiro2 = UFPR04.classes
            y_predicao2 = modelo.predict(UFPR04).argmax(axis=1)

            y_verdadeiro3 = UFPR05.classes
            y_predicao3 = modelo.predict(UFPR05).argmax(axis=1)
            
        elif 'UFPR04' in modelo_nome: 
            y_verdadeiro1 = PUC.classes
            y_predicao1 = modelo.predict(PUC).argmax(axis=1)

            y_verdadeiro2 = UFPR04.classes
            y_predicao2 = modelo.predict(UFPR04).argmax(axis=1)
            
            y_verdadeiro3 = UFPR05.classes
            y_predicao3 = modelo.predict(UFPR05).argmax(axis=1)

        elif 'UFPR05' in modelo_nome:
            y_verdadeiro1 = PUC.classes
            y_predicao1 = modelo.predict(PUC).argmax(axis=1)

            y_verdadeiro2 = UFPR04.classes
            y_predicao2 = modelo.predict(UFPR04).argmax(axis=1)

            y_verdadeiro3 = UFPR05.classes
            y_predicao3 = modelo.predict(UFPR05).argmax(axis=1)
        
        labels = ['Empty', 'Occupied'] 
        save_path = os.path.join(f'Resultados/Matriz_de_confusao/{modelo_nome}/', f'{modelo_nome}_vs_PUC_matriz_de_confusao.png')
        titulo = f'Matriz de Confusão - {modelo_nome} vs PUC'
        plot_confusion_matrix(y_verdadeiro1, y_predicao1, labels, save_path, titulo)

        save_path = os.path.join(f'Resultados/Matriz_de_confusao/{modelo_nome}/', f'{modelo_nome}_vs_UFPR04_matriz_de_confusao.png')
        titulo = f'Matriz de Confusão - {modelo_nome} vs UFPR04'
        plot_confusion_matrix(y_verdadeiro2, y_predicao2, labels, save_path, titulo)

        save_path = os.path.join(f'Resultados/Matriz_de_confusao/{modelo_nome}/', f'{modelo_nome}_vs_UFPR05_matriz_de_confusao.png')
        titulo = f'Matriz de Confusão - {modelo_nome} vs UFPR05'
        plot_confusion_matrix(y_verdadeiro3, y_predicao3, labels, save_path, titulo)

# Exemplo: 
predict_e_matriz_de_confusao()




