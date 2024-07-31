import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf

# Suponha que seus dados já foram carregados em X_train, X_val e X_test
# Certifique-se de que X_train, X_val e X_test têm o formato correto (n_samples, 256, 256, 3)

# Normalizando os dados para o intervalo [0, 1], se necessário

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocessamento import preprocessamento

csv_file = 'Datasets_csv/df_PUC.csv'
train, teste, val, _, _, _ = preprocessamento(csv_file, 0.6, 0.2, 0.2, True)

X_train, _ = next(train)
X_test, _ = next(teste)
X_val, _ = next(val)

X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Definindo a classe Autoencoder
class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),  # Flatten a entrada para um vetor unidimensional
            layers.Dense(latent_dim, activation='relu')  # Camada densa para a codificação
        ], name='encoder')
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(np.prod(shape), activation='sigmoid'),  # Camada densa para a decodificação
            layers.Reshape(shape)  # Remodela de volta para o formato original da entrada
        ], name='decoder')

    def call(self, x):
        encoded = self.encoder(x)  # Codifica a entrada
        decoded = self.decoder(encoded)  # Decodifica a saída codificada
        return decoded

# Definindo parâmetros
latent_dim = 64  # Dimensão da codificação
shape = X_train.shape[1:]  # Formato das imagens de entrada

# Instanciando o Autoencoder
autoencoder = Autoencoder(latent_dim, shape)

# Compilando o modelo
autoencoder.compile(optimizer='adam', loss='mse')  # Usando MSE como loss para reconstrução

# Treinando o modelo com os dados de treino e validação
history = autoencoder.fit(X_train, X_train, epochs=10, batch_size=128, shuffle=True, validation_data=(X_val, X_val))

# Passando algumas imagens de teste pelo autoencoder para reconstruí-las
decoded_images = autoencoder.predict(X_test)

# Função para plotar imagens originais e reconstruídas
def plot_images(original, reconstructed, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Imagem original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i])
        plt.title("Original")
        plt.axis('off')
        
        # Imagem reconstruída
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i])
        plt.title("Reconstruída")
        plt.axis('off')
    plt.show()

# Plotando algumas imagens originais e reconstruídas
plot_images(X_test, decoded_images)
