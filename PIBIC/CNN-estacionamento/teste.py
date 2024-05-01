import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv('datasets\df_conjunto_5k.csv')

# Divida o DataFrame em conjuntos de treino e teste
treino, teste = train_test_split(dataframe, test_size=0.2, random_state=42)
nome_classe = ['ocupada', 'vaga']

plt.figure(figsize=(15, 15))
for i in range(9):
    caminho_imagem = treino.iloc[i]['caminho_imagem']
    imagem = Image.open(caminho_imagem)
    plt.subplot(3, 3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imagem)
    plt.xlabel(treino.iloc[i]['classe'], fontsize=25)
plt.show()

# Defina os parâmetros da imagem
img_width, img_height = 64, 64
batch_size = 32

# Inicialize os geradores de dados com normalização
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=treino,
    x_col='caminho_imagem',
    y_col='classe',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=teste,
    x_col='caminho_imagem',
    y_col='classe',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

modelo_pucpr = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilando o modelo
modelo_pucpr.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Checkpoint
checkpoint_path = 'weights-improvement-{epoch:02d}-{val_accuracy:.2f}.weights.h5'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# Treinando o modelo
modelo_pucpr.fit(
    train_generator,
    epochs=10,
    callbacks=cp_callback,
    validation_data=test_generator
)

# Avaliando o modelo
loss, accuracy = modelo_pucpr.evaluate(test_generator)
print(f'Perda de teste: {loss:.4f}, Precisão de teste: {accuracy:.4f}')

modelo_pucpr.save("modelo_pucpr.keras")
