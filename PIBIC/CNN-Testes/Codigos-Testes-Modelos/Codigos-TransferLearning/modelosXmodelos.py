import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Função para carregar modelos e pesos
def carregar_modelo(caminho_modelo, caminho_pesos):
    modelo = keras.models.load_model(caminho_modelo)
    modelo.load_weights(caminho_pesos)
    return modelo

modelos = {
    "PUCPR": carregar_modelo("PIBIC/CNN-Testes/Modelos-keras/PUCPR_mobilenetv3.keras", 
                             'PIBIC/CNN-Testes/weights-finais/PUCPR_mobilenetv3.h5'),
    
    "UFPR04": carregar_modelo("PIBIC/CNN-Testes/Modelos-keras/UFPR04_mobilenetv3.keras", 
                              'PIBIC/CNN-Testes/weights-finais/UFPR04_mobilenetv3.h5'),

    "UFPR05": carregar_modelo("PIBIC/CNN-Testes/Modelos-keras/UFPR05_mobilenetv3.keras", 
                              'PIBIC/CNN-Testes/weights-finais/UFPR05_mobilenetv3.h5'),

    "PUCPRConv2": carregar_modelo("PIBIC/CNN-Testes/Modelos-keras/PUCPR_mobilenetv3_2.keras", 
                                  'PIBIC/CNN-Testes/weights-finais/PUCPR_mobilenetv3_2.h5'),

    "UFPR04Conv2": carregar_modelo("PIBIC/CNN-Testes/Modelos-keras/UFPR04_mobilenetv3_2.keras", 
                                   'PIBIC/CNN-Testes/weights-finais/UFPR04_mobilenetv3_2.h5'),

    "UFPR05Conv2": carregar_modelo("PIBIC/CNN-Testes/Modelos-keras/UFPR05_mobilenetv3_2.keras", 
                                   'PIBIC/CNN-Testes/weights-finais/UFPR05_mobilenetv3_2.h5')
}

# Carregando os DataFrames
datasets = {
    "PUCPR": pd.read_csv('PIBIC/CNN-Testes/Datasets/df_PUCPR_Teste.csv'),
    "UFPR04": pd.read_csv('PIBIC/CNN-Testes/Datasets/df_UFPR04.csv'),
    "UFPR05": pd.read_csv('PIBIC/CNN-Testes/Datasets/df_UFPR05.csv')
}

def modeloBaseContraTeste(modeloBase, dataframeTeste, nomeConjuntoTeste): 
    img_width, img_height = 224, 224
    batch_size = 32

    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    dataframeGerador = datagen.flow_from_dataframe(
        dataframe=dataframeTeste,
        x_col='caminho_imagem',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )

    predicoes = modeloBase.predict(dataframeGerador)
    limiar = 0.5
    classes_preditas_numericas = (predicoes > limiar).astype(int)
    classes_preditas = ['Empty' if pred == 1 else 'Occupied' for pred in classes_preditas_numericas]
    dataframeTeste['classe_predita'] = classes_preditas

    print(dataframeTeste.head())  
    
    acertos = (dataframeTeste['classe_predita'] == dataframeTeste['classe']).sum()
    total = len(dataframeTeste)
    porcentagem_acertos = (acertos / total) * 100
    print(f"{nomeConjuntoTeste} - Porcentagem de acerto: {porcentagem_acertos:.2f}%")

    return porcentagem_acertos

resultados = pd.DataFrame(columns=datasets.keys(), index=modelos.keys())
print("Tabela Final: ")
for nome_modelo, modelo in modelos.items():
    for nome_dataset, dataset in datasets.items():
        porcentagem_acertos = modeloBaseContraTeste(modelo, dataset, f'{nome_modelo} X {nome_dataset}')
        resultados.at[nome_modelo, nome_dataset] = f"{porcentagem_acertos:.2f}%"

print(resultados)