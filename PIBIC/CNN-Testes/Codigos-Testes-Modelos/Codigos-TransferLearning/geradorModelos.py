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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import datasets, layers, models, losses


os.environ["CUDNN_PATH"] = "/home/lucas/Documents/PIBIC/CNN-testes/.venv/lib/python3.11/site-packages/nvidia/cudnn"
os.environ["LD_LIBRARY_PATH"] = "$CUDNN_PATH:$LD_LIBRARY_PATH"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda/"

#Plot a training history
def plot_history(history):
  print(history.history.keys())
  # summarize history for accuracy
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()


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

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    treino_datagen = ImageDataGenerator(rescale=1./127.5)
    validacao_datagen = ImageDataGenerator(rescale=1./127.5)
    teste_dategen = ImageDataGenerator(rescale=1./127.5)

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

    class_names = treino_gerador.classes
    num_classes = len(class_names)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        ])

    base_modelo = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

    image_batch, label_batch = next(iter(treino_gerador))
    feature_batch = base_modelo(image_batch)

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    base_modelo.trainable = False 
    print("Number of layers in the base model: ", len(base_modelo.layers))

    print("Camadas treináveis:")
    for layer in base_modelo.layers:
        print(layer.name, layer.trainable)

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)

    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    model = Sequential([
        base_modelo,
        GlobalAveragePooling2D(),
        Dense(1, activation='sigmoid')
    ])

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    model.summary()

    """checkpoint_path = 'PIBIC/CNN-Testes/weights/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.weights.h5'
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, 
                                  monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)"""
    history = model.fit(
        treino_gerador,
        epochs=10,
        validation_data=validacao_gerador
    )

    plot_history(history)

    perda, precisao = model.evaluate(teste_gerador)
    print(f'{nomeModelo} - Perda de teste: {perda:.4f}, Precisão de teste: {precisao:.4f}')

    model.save(f"PIBIC/CNN-Testes/Modelos-keras/{nomeModelo}_mobilenetv3_2.keras")
    model.save_weights(f"PIBIC/CNN-Testes/weights-finais/{nomeModelo}_mobilenetv3_2.h5")
    
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

    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

    treino_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    validacao_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    teste_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
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

    teste_gerador = teste_datagen.flow_from_dataframe(
        dataframe=teste,
        x_col='caminho_imagem',
        y_col='classe',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        ])

    base_modelo = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

    image_batch, label_batch = next(iter(treino_gerador))
    feature_batch = base_modelo(image_batch)

    base_modelo.trainable = False 

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)

    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_modelo(x, training=False)
    x = global_average_layer(x)
    #x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    

    #checkpoint
    checkpoint_path = 'PIBIC/CNN-Testes/weights/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.weights.h5'
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, 
                                  monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

    history = model.fit(
        treino_gerador,
        epochs=10,
        callbacks=[cp_callback],
        validation_data=validacao_gerador
    )

    print("Numero de camadas no modelo: ", len(base_modelo.layers))

    # Primeiro, defina todas as camadas para não treináveis
    for layer in base_modelo.layers:
        layer.trainable = False

    # Encontre todas as camadas convolucionais no modelo
    conv_layers = [layer for layer in base_modelo.layers if isinstance(layer, Conv2D)]

    # Em seguida, defina as duas últimas camadas convolucionais para treináveis
    for layer in conv_layers[-2:]:
        layer.trainable = True

    # Confira se as camadas estão agora congeladas ou não
    print("Camdas treináveis depois do fine-tuning:")
    for layer in base_modelo.layers:
        print(layer.name, layer.trainable)


    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
                metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])
    
    model.summary()
    len(model.trainable_variables)

    history_fine = model.fit(treino_gerador,
                            epochs=10,
                            callbacks=[cp_callback],
                            validation_data=validacao_gerador)
    
    plot_history(history_fine)


    perda, precisao = model.evaluate(teste_gerador)
    print(f'{nomeModelo} - Perda de teste: {perda:.4f}, Precisão de teste: {precisao:.4f}')

    model.save(f"PIBIC/CNN-Testes/Modelos-keras/{nomeModelo}_mobilenetv3_2.keras")
    model.save_weights(f"PIBIC/CNN-Testes/weights-finais/{nomeModelo}_mobilenetv3_2.h5")


"""geradorModelosConvMobileNetV3('PIBIC/CNN-Testes/Datasets/df_PUC.csv', 'PUCPR')
geradorModelosConvMobileNetV3('PIBIC/CNN-Testes/Datasets/df_UFPR04.csv', 'UFPR04')
geradorModelosConvMobileNetV3('PIBIC/CNN-Testes/Datasets/df_UFPR05.csv', 'UFPR05')"""
print("\n\n\n\n")
geradorModelosMobileNetV3('PIBIC/CNN-Testes/Datasets/df_PUC.csv', 'PUCPR')
geradorModelosMobileNetV3('PIBIC/CNN-Testes/Datasets/df_UFPR04.csv', 'UFPR04')
geradorModelosMobileNetV3('PIBIC/CNN-Testes/Datasets/df_UFPR05.csv', 'UFPR05')
