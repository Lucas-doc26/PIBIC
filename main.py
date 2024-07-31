import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from preprocessamento import preprocessamento

# Carregar dados usando preprocessamento
treino_gerador, validacao_gerador, teste_gerador, _, _, _ = preprocessamento('Datasets_csv\df_PUC.csv', 0.6, 0.2, 0.2, True)

input_img = Input(shape=(256, 256, 3))  # Ajuste conforme o tamanho da imagem

# Codificador
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decodificador
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Ajustar o número de épocas conforme necessário
batch_size = 32
epochs = 5

# Calcular steps_per_epoch e validation_steps
steps_per_epoch = len(treino_gerador) // batch_size
validation_steps = len(validacao_gerador) // batch_size

# Treinar o autoencoder
autoencoder.fit(treino_gerador,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=validacao_gerador,
                validation_steps=validation_steps)

# Escolher algumas imagens de teste
test_images, _ = next(teste_gerador)
decoded_imgs = autoencoder.predict(test_images)

# Escala para o intervalo de 0 a 255
decoded_imgs_scaled = (decoded_imgs * 255).astype(np.uint8)

# Visualizar as imagens originais e reconstruídas
n = 10  # Número de imagens a serem exibidas
plt.figure(figsize=(20, 4))
for i in range(n):
    # Mostrar a imagem original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_images[i].astype(np.uint8))
    plt.title("Original")
    plt.axis('off')

    # Mostrar a imagem reconstruída
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs_scaled[i])
    plt.title("Reconstructed")
    plt.axis('off')
plt.tight_layout()
plt.show()
