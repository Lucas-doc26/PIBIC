import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt

modelo_pucpr = keras.models.load_model("modelo_pucpr.keras")

dataframe_ufpr05 = pd.read_csv('Datasets/df_ufpr05.csv')
print(dataframe_ufpr05.head())

img_width, img_height = 64, 64
batch_size = 32

ufpr05_datagen = ImageDataGenerator(rescale=1./255)

ufpr05_gerador = ufpr05_datagen.flow_from_dataframe(
    dataframe=dataframe_ufpr05,
    x_col='caminho_imagem',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)


predicoes = modelo_pucpr.predict(ufpr05_gerador)
limiar = 0.5
classes_preditas_numericas = (predicoes > limiar).astype(int)
classes_preditas = ['ocupada' if pred == 0 else 'vaga' for pred in classes_preditas_numericas]
dataframe_ufpr05['classe_predita'] = classes_preditas

plt.figure(figsize=(15, 15))
plt.title("Dataset UFPR05", fontsize='25')
for i in range(9):
    caminho_imagem = dataframe_ufpr05.iloc[i]['caminho_imagem']
    imagem = Image.open(caminho_imagem)
    plt.subplot(3, 3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imagem)
    plt.xlabel(dataframe_ufpr05.iloc[i]["classe_predita"], fontsize=14)
plt.show()

acertos = (dataframe_ufpr05['classe_predita'] == dataframe_ufpr05['classe']).sum()
total = len(dataframe_ufpr05)
porcentagem_acertos = (acertos / total) * 100
print(f"Porcentagem de acertos: {porcentagem_acertos:.2f}%")

dataframe_ufpr05.to_csv("Datasets/df_ufpr05.csv", index=False)





