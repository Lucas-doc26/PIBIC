import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2


dataframe = pd.read_csv('Datasets/df_PUC.csv')

# divisao - 60%/20%/20
treino, teste = train_test_split(dataframe, test_size=0.2, random_state=42)
treino, validacao = train_test_split(treino, test_size=0.25, random_state=42)

nome_classe = ['ocupada', 'vaga']
print(treino.head())
plt.figure(figsize=(15, 15))
plt.title("Dataset PUCPR", fontsize=25)
plt.grid(False)
for i in range(9):
    try:
        caminho_imagem = treino.iloc[i]['caminho_imagem']
        imagem = Image.open(caminho_imagem)
        plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imagem)
        plt.xlabel(treino.iloc[i]['classe'], fontsize=25)
    except FileNotFoundError:
        print(f"Erro ao abrir o arquivo: {caminho_imagem}")
plt.show()

#parâmetros da imagem
img_width, img_height = 64, 64
batch_size = 32

treino_datagen = ImageDataGenerator(rescale=1./255)
validacao_datagen = ImageDataGenerator(rescale=1./255)
teste_dategen = ImageDataGenerator(rescale=1./255)

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
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

teste_generator = teste_dategen.flow_from_dataframe(
    dataframe=teste,
    x_col='caminho_imagem',
    y_col='classe',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Definir arquitetura do modelo
from tensorflow.keras.layers import Dropout

modelo_pucpr = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),  # Dropout de 25% após a camada de pooling
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),  # Dropout de 25% após a camada de pooling
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    #Dropout(0.5),  # Dropout de 50% após a camada densa
    Dense(1, activation='sigmoid')
])

# Compilar o modelo
modelo_pucpr.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#checkpoint
checkpoint_path = 'weights/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.weights.h5'
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# Treinar o modelo com validação
history = modelo_pucpr.fit(
    treino_gerador,
    epochs=10,
    callbacks=[cp_callback],
    validation_data=validacao_gerador
)

perca, precisao = modelo_pucpr.evaluate(teste_generator)
print(f'Perda de teste: {perca:.4f}, Precisão de teste: {precisao:.4f}')

modelo_pucpr.save("modelo_pucpr.keras")
modelo_pucpr.save_weights("weights_pucpr.weights.h5")