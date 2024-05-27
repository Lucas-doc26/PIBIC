import pandas as pd
from tensorflow import keras
import numpy as np
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def carregar_modelo_e_pesos(nome_modelo, nome_pesos):
    modelo = keras.models.load_model(nome_modelo)
    modelo.load_weights(nome_pesos)
    return modelo

def carregar_dataframe(caminho_csv):
    return pd.read_csv(caminho_csv)

def fazer_previsoes_e_calcular_acertos(modelo, dataframe):
    # Carregar e pré-processar as imagens
    imagens = []
    for caminho_imagem in dataframe['caminho_imagem']:
        img = Image.open(caminho_imagem).convert('RGB')
        img = img.resize((224, 224))  # Redimensionar para o tamanho esperado pelo modelo
        img = np.array(img)  # Converter para array numpy
        img = img / 255.0  # Normalização
        img = np.expand_dims(img, axis=0)  # Adicionar dimensão do batch
        imagens.append(img)

    # Concatenar as imagens em um único array
    imagens = np.vstack(imagens)

    # Fazer previsões
    predicoes = modelo.predict(imagens)

    # Converter as previsões em classes
    classes_preditas = ['Empty' if pred < 0.5 else 'Occupied' for pred in predicoes]

    # Adicionar as classes preditas ao DataFrame
    dataframe['classe_predita'] = classes_preditas

    # Calcular a porcentagem de acertos
    acertos = (dataframe['classe_predita'] == dataframe['classe']).sum()
    total = len(dataframe)
    porcentagem_acertos = (acertos / total) * 100

    return porcentagem_acertos

# Carregar os modelos
modelo_PUCPRConv2 = carregar_modelo_e_pesos("PIBIC/CNN-Testes/Modelos-keras/PUCPR_mobilenetv3_2.keras", "PIBIC/CNN-Testes/weights-finais/PUCPR_mobilenetv3_2.h5")
modelo_PUCPR = carregar_modelo_e_pesos("PIBIC/CNN-Testes/Modelos-keras/PUCPR_mobilenetv3.keras", "PIBIC/CNN-Testes/weights-finais/PUCPR_mobilenetv3.h5")
modelo_UFPR04Conv2 = carregar_modelo_e_pesos("PIBIC/CNN-Testes/Modelos-keras/UFPR04_mobilenetv3_2.keras", "PIBIC/CNN-Testes/weights-finais/UFPR04_mobilenetv3_2.h5")
modelo_UFPR04 = carregar_modelo_e_pesos("PIBIC/CNN-Testes/Modelos-keras/UFPR04_mobilenetv3.keras", "PIBIC/CNN-Testes/weights-finais/UFPR04_mobilenetv3.h5")
modelo_UFPR05Conv2 = carregar_modelo_e_pesos("PIBIC/CNN-Testes/Modelos-keras/UFPR05_mobilenetv3_2.keras", "PIBIC/CNN-Testes/weights-finais/UFPR05_mobilenetv3_2.h5")
modelo_UFPR05 = carregar_modelo_e_pesos("PIBIC/CNN-Testes/Modelos-keras/UFPR05_mobilenetv3.keras", "PIBIC/CNN-Testes/weights-finais/UFPR05_mobilenetv3.h5")

# Carregar os DataFrames
dataframe_PUCPR = carregar_dataframe('PIBIC/CNN-Testes/Datasets/df_PUCPR_Teste.csv')
dataframe_UFPR04 = carregar_dataframe('PIBIC/CNN-Testes/Datasets/df_UFPR04.csv')
dataframe_UFPR05 = carregar_dataframe('PIBIC/CNN-Testes/Datasets/df_UFPR05.csv')

# Fazer previsões e calcular acertos para todos os modelos em todos os DataFrames
resultados = {}

for nome_modelo, modelo in [("PUCPRConv2", modelo_PUCPRConv2), ("PUCPR", modelo_PUCPR), ("UFPR04Conv2", modelo_UFPR04Conv2), ("UFPR04", modelo_UFPR04), ("UFPR05Conv2", modelo_UFPR05Conv2), ("UFPR05", modelo_UFPR05)]:
    for nome_dataframe, dataframe in [("PUCPR", dataframe_PUCPR), ("UFPR04", dataframe_UFPR04), ("UFPR05", dataframe_UFPR05)]:
        porcentagem_acertos = fazer_previsoes_e_calcular_acertos(modelo, dataframe)
        resultados[(nome_modelo, nome_dataframe)] = porcentagem_acertos

# Imprimir os resultados
for (nome_modelo, nome_dataframe), porcentagem_acertos in resultados.items():
    print(f"Porcentagem de acertos do modelo {nome_modelo} no dataframe {nome_dataframe}: {porcentagem_acertos:.2f}%")
