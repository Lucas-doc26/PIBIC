import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV3Small
from sklearn.model_selection import train_test_split

os.environ["CUDNN_PATH"] = "/home/lucas/Documents/PIBIC/CNN-testes/.venv/lib/python3.11/site-packages/nvidia/cudnn"
os.environ["LD_LIBRARY_PATH"] = "$CUDNN_PATH:$LD_LIBRARY_PATH"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda/"

def geradorModelosMobileNetV3(caminho:str, nomeModelo:str):

    """
    Gerador de modelo MobilenetV3. 

    :param caminho: string
    :param nomeModelo: string

    :return Salva o modelo como nomeModelo.keras :
    """
    dataframe = pd.read_csv(caminho)
    treino, teste = train_test_split(dataframe, test_size=0.2, random_state=42)
    treino, validacao = train_test_split(treino, test_size=0.25, random_state=42)

    img_width, img_height = 224, 224  # Recomendado para MobileNetV3
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

    teste_gerador = teste_dategen.flow_from_dataframe(
        dataframe=teste,
        x_col='caminho_imagem',
        y_col='classe',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    base_modelo = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

    base_modelo.trainable = False

    # Adicionar as camadas Fully-Connected
    model = Sequential([
        base_modelo,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(1, activation='sigmoid')
    ])

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    model.summary(show_trainable = True)
    

    history = model.fit(
        treino_gerador,
        epochs=10,
        validation_data=validacao_gerador
    )

    perca, precisao = model.evaluate(teste_gerador)
    print(f'{nomeModelo} - Perda de teste: {perca:.4f}, Precisão de teste: {precisao:.4f}')

    model.save(f"PIBIC/CNN-Testes/Modelos-keras/{nomeModelo}_mobilenetv3.keras")
    model.save_weights(f"PIBIC/CNN-Testes/weights-finais/{nomeModelo}_mobilenetv3.h5")

def geradorModelosConvMobileNetV3(caminho:str, nomeModelo:str):

    """
    Gerador de modelo MobilenetV3. 

    :param caminho: string
    :param nomeModelo: string

    :return Salva o modelo como nomeModelo.keras :
    """
    dataframe = pd.read_csv(caminho)
    treino, teste = train_test_split(dataframe, test_size=0.2, random_state=42)
    treino, validacao = train_test_split(treino, test_size=0.25, random_state=42)

    img_width, img_height = 224, 224  # Recomendado para MobileNetV3
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
        target_sizgeradorModelosConvMobileNetV3e=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    teste_gerador = teste_dategen.flow_from_dataframe(
        dataframe=teste,
        x_col='caminho_imagem',
        y_col='classe',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    base_modelo = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

    # Fine-tune from this layer onwards
    fine_tune_at = 2

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_modelo.layers[:-fine_tune_at]:
        layer.trainable = False

    # Adicionar as camadas Fully-Connected
    model = Sequential([
        base_modelo,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.25),  # Dropout de 25% após a camada de pooling
        Dense(1, activation='sigmoid')
    ])

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    model.summary(show_trainable = True)
    

    history = model.fit(
        treino_gerador,
        epochs=10,
        validation_data=validacao_gerador
    )

    perda, precisao = model.evaluate(teste_gerador)
    print(f'{nomeModelo} - Perda de teste: {perda:.4f}, Precisão de teste: {precisao:.4f}')

    model.save(f"{nomeModelo}_mobilenetv3.h5")

geradorModelosMobileNetV3('PIBIC/CNN-Testes/Datasets/df_PUC.csv', 'PUCPR')
print("\n\n\n\n")
geradorModelosConvMobileNetV3('PIBIC/CNN-Testes/Datasets/df_PUC.csv', 'PUCPR')
#geradorModelosMobileNetV3('PIBIC/CNN-Testes/Datasets/df_UFPR04.csv', 'UFPR04')
#geradorModelosMobileNetV3('PIBIC/CNN-Testes/Datasets/df_UFPR05.csv', 'UFPR05')
