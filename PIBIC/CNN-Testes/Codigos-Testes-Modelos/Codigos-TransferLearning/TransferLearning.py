import os
import numpy as np
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

def plot_sidebyside(img_list,titles,colormap=None,figsize=(12,6)):
  n = len(img_list)
  figure, axis = plt.subplots(1, n, figsize=figsize)
    
  for i in range(n):         
    axis[i].imshow(img_list[i], cmap=colormap)
    axis[i].set_title(titles[i])
    axis[i].axis('off')
  plt.show()

def plot_dataset(ds, lbls_name):
  N_SAMPLES = 10
  for i in range(5):
    for x,y in ds.take(1):    
      
      x = x.numpy()
      x = np.squeeze(x)      
      y = y.numpy()
      plot_sidebyside(x[:N_SAMPLES],
                      y[:N_SAMPLES],'gray')

def plot_history(history):
  print(history.history.keys())
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()


def create_model(trainable_conv:int):
  img_width, img_height = 224, 224 
  base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3), pooling='avg')
  for layer in base_model.layers:
    layer.trainable = False

  if trainable_conv == 2:
    conv_layers = [layer for layer in base_model.layers if isinstance(layer, Conv2D)]
    for layer in conv_layers[-trainable_conv:]:
        layer.trainable = True

  model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='sigmoid'),
    Dense(128, activation='sigmoid'),
    Dense(2, activation='softmax')
  ])

  return model 

def preprocess(caminho:str):

  dataframe = pd.read_csv(caminho)
  treino, teste = train_test_split(dataframe, test_size=0.2, random_state=42)
  treino, validacao = train_test_split(treino, test_size=0.25, random_state=42)

  img_width, img_height = 224, 224
  batch_size = 32

  preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
  treino_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
  validacao_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
  teste_dategen = ImageDataGenerator(preprocessing_function=preprocess_input)
  
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
  return treino_gerador, validacao_gerador, teste_gerador

def preprocess_teste(caminho:str):

  img_width, img_height = 224, 224
  batch_size = 32

  dataframe = pd.read_csv(caminho)
  preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
  datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
  dataframe_gerador = datagen.flow_from_dataframe(
          dataframe=dataframe,
          x_col='caminho_imagem',
          y_col='classe',
          target_size=(img_width, img_height),
          batch_size=batch_size,
          class_mode='binary'
      )
  return dataframe_gerador

def teste_modelos(caminho:str, num:int):
  treino, validacao, teste = preprocess(caminho)
  model = create_model(num)
  model.summary(show_trainable=True)

  base_learning_rate = 0.0001
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
  
  model.fit(treino,
                epochs=10,
                validation_data=validacao)

  if caminho.__contains__('PUC'):
    UFPR04 = preprocess_teste('PIBIC/CNN-Testes/Datasets/df_UFPR04.csv')
    UFPR05 = preprocess_teste('PIBIC/CNN-Testes/Datasets/df_UFPR05.csv')
    print(f"PUC X PUC - precissão de {model.evaluate(teste)[1]:.3f} ")
    print(f"PUC X UFPR04 - precissão de {model.evaluate(UFPR04)[1]:.3f} ")
    print(f"PUC X UFPR05 - precissão de {model.evaluate(UFPR05)[1]:.3f} ")
  elif caminho.__contains__('UFPR04'):
    PUC = preprocess_teste('PIBIC/CNN-Testes/Datasets/df_PUC.csv')
    UFPR05 = preprocess_teste('PIBIC/CNN-Testes/Datasets/df_UFPR05.csv')
    print(f"UFPR04 X PUC - precissão de {model.evaluate(PUC)[1]:.3f} ")
    print(f"UFPR04 X UFPR04 - precissão de {model.evaluate(teste)[1]:.3f} ")
    print(f"UFPR04 X UFPR05 - precissão de {model.evaluate(UFPR05)[1]:.3f} ")
  else:
    PUC = preprocess_teste('PIBIC/CNN-Testes/Datasets/df_PUC.csv')
    UFPR04 = preprocess_teste('PIBIC/CNN-Testes/Datasets/df_UFPR04.csv')
    print(f"UFPR05 X PUC - precissão de {model.evaluate(PUC)[1]:.3f} ")
    print(f"UFPR05 X UFPR04 - precissão de {model.evaluate(UFPR04)[1]:.3f} ")
    print(f"UFPR05 X UFPR05 - precissão de {model.evaluate(teste)[1]:.3f} ")

print("Modelos sem o Fine Tuning:\n")
teste_modelos('PIBIC/CNN-Testes/Datasets/df_PUC.csv', 0)
teste_modelos('PIBIC/CNN-Testes/Datasets/df_UFPR04.csv', 0)
teste_modelos('PIBIC/CNN-Testes/Datasets/df_UFPR05.csv', 0)

print("\nModelos com o Fine Tuning:\n")
teste_modelos('PIBIC/CNN-Testes/Datasets/df_PUC.csv', 2)
teste_modelos('PIBIC/CNN-Testes/Datasets/df_UFPR04.csv', 2)
teste_modelos('PIBIC/CNN-Testes/Datasets/df_UFPR05.csv', 2)