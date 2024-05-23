from cProfile import label
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt

modeloPUCPR = keras.models.load_model("PIBIC/CNN-Testes/Modelos-keras/PUCPR_mobilenetv3.keras")
modeloPUCPR.load_weights('PIBIC/CNN-Testes/weights-finais/PUCPR_mobilenetv3.h5')

modeloPUCPRConv2 = keras.models.load_model("PIBIC/CNN-Testes/Modelos-keras/PUCPR_2Conv_mobilenetv3.keras")
modeloPUCPRConv2 = keras.models.load_model("PIBIC/CNN-Testes/Modelos-keras/PUCPR_2Conv_mobilenetv3.keras")


modeloUFPR04 = keras.models.load_model("PIBIC/CNN-Testes/Modelos-keras/UFPR04_mobilenetv3.keras")
modeloUFPR04.load_weights('PIBIC/CNN-Testes/weights-finais/UFPR04_mobilenetv3.h5')

modeloUFPR05 = keras.models.load_model("PIBIC/CNN-Testes/Modelos-keras/UFPR05_mobilenetv3.keras")
modeloUFPR05.load_weights('PIBIC/CNN-Testes/weights-finais/UFPR05_mobilenetv3.h5')

dataframePUCPR = pd.read_csv('PIBIC/CNN-Testes/Datasets/df_PUCPR_Teste.csv')
dataframeUFPR04 = pd.read_csv('PIBIC/CNN-Testes/Datasets/df_UFPR04.csv')
dataframeUFPR05 = pd.read_csv('PIBIC/CNN-Testes/Datasets/df_UFPR05.csv')

def modeloBaseContraTeste(modeloBase:keras, dataframeTeste:pd.DataFrame, nomeConjuntoTeste:str): 
    img_width, img_height = 224, 224
    batch_size = 32

    datagen = ImageDataGenerator(rescale=1./255)
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
    classes_preditas = ['Empty' if pred == 0 else 'Occupied' for pred in classes_preditas_numericas]

    dataframeTeste['classe_predita'] = classes_preditas

    acertos = (dataframeTeste['classe_predita'] == dataframeTeste['classe']).sum()
    total = len(dataframeTeste)
    porcentagem_acertos = (acertos / total) * 100
    print(f"{nomeConjuntoTeste} - Porcentagem de acerto: {porcentagem_acertos:.2f}%")

    """plt.figure(figsize=(15, 15))
    plt.title(f"{nomeConjuntoTeste}", fontsize='25')
    for i in range(9):
        caminho_imagem = dataframeTeste.iloc[i]['caminho_imagem']
        imagem = Image.open(caminho_imagem)
        plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imagem)
        plt.xlabel(dataframeTeste.iloc[i]["classe_predita"], fontsize=14)
    plt.xticks([])
    plt.yticks([])
    plt.figtext(0.5, 0.01, f"Porcentagem de acertos: {porcentagem_acertos:.2f} %", ha='center', fontsize=14)
    plt.show()"""

    return porcentagem_acertos

"""modeloBaseContraTeste(modeloPUCPR, dataframePUCPR, 'PUCPR X PUCPR')
modeloBaseContraTeste(modeloPUCPR, dataframeUFPR04, 'PUCPR X UFPR04')
modeloBaseContraTeste(modeloPUCPR, dataframeUFPR05, 'PUCPR X UFPR05')

modeloBaseContraTeste(modeloUFPR04, dataframePUCPR, 'UFPR04 x PUCPR')
modeloBaseContraTeste(modeloUFPR04, dataframeUFPR04, 'UFPR04 x UFPR04')
modeloBaseContraTeste(modeloUFPR04, dataframeUFPR05, 'UFPR04 x UFPR05')

modeloBaseContraTeste(modeloUFPR05, dataframePUCPR, 'UFPR05 x PUCPR')
modeloBaseContraTeste(modeloUFPR05, dataframeUFPR04, 'UFPR05 x UFPR04')
modeloBaseContraTeste(modeloUFPR05, dataframeUFPR05, 'UFPR05 x UFPR05')"""

"""tabela = [[None for _ in range(3)] for _ in range(3)]

tabela[0][0] = modeloBaseContraTeste(modeloPUCPR, dataframePUCPR, 'PUCPR X PUCPR')
tabela[0][1] = modeloBaseContraTeste(modeloPUCPR, dataframeUFPR04, 'PUCPR X UFPR04')
tabela[0][2] = modeloBaseContraTeste(modeloPUCPR, dataframeUFPR05, 'PUCPR X UFPR05')

tabela[1][0] = modeloBaseContraTeste(modeloUFPR04, dataframePUCPR, 'UFPR04 X PUCPR')
tabela[1][1] = modeloBaseContraTeste(modeloUFPR04, dataframeUFPR04, 'UFPR04 X UFPR04')
tabela[1][2] = modeloBaseContraTeste(modeloUFPR04, dataframeUFPR05, 'UFPR04 X UFPR05')

tabela[2][0] = modeloBaseContraTeste(modeloUFPR05, dataframePUCPR, 'UFPR05 X PUCPR')
tabela[2][1] = modeloBaseContraTeste(modeloUFPR05, dataframeUFPR04, 'UFPR05 X UFPR04')
tabela[2][2] = modeloBaseContraTeste(modeloUFPR05, dataframeUFPR05, 'UFPR05 X UFPR05')

print('PUCPR, UFPR04, UFPR05')
for i in range(len(tabela)):
    print(tabela[i])
"""

modeloBaseContraTeste(modeloPUCPR, dataframePUCPR, 'PUCPR X PUCPR')
modeloBaseContraTeste(modeloPUCPR, dataframeUFPR04, 'PUCPR X UFPR04')
modeloBaseContraTeste(modeloPUCPR, dataframeUFPR05, 'PUCPR X UFPR05')

modeloBaseContraTeste(modeloPUCPRConv2, dataframePUCPR, 'PUCPR Conv2 X PUCPR')
modeloBaseContraTeste(modeloPUCPRConv2, dataframeUFPR04, 'PUCPR Conv2 X UFPR04')
modeloBaseContraTeste(modeloPUCPRConv2, dataframeUFPR05, 'PUCPR Conv2 X UFPR05')
