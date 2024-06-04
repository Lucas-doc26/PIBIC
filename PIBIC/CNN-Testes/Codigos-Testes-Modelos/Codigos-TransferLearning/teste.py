import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import metrics

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_datasets as tfds

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


dataset = pd.read_csv('PIBIC/CNN-Testes/Datasets/df_PUC.csv')

treino, teste = train_test_split(dataset, test_size=0.2, random_state=42)
treino, validacao = train_test_split(treino, test_size=0.25, random_state=42)

img_width, img_height = 224, 224  # Recomendado para MobileNetV3
batch_size = 32
input_shape_ = (224,224,3)

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

classes = treino_gerador.classes
num_classes = len(classes)
print(classes)
classes = np.array(classes)

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

model = models.Sequential()

#32 layers of size 3x3 and Relu Activation
model.add(layers.Rescaling(1./255,input_shape=input_shape_))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
#Max Pooling of Size (2x2)
model.add(layers.MaxPooling2D((2, 2)))


#64 layers of size 3x3 and Relu Activation
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#Max Pooling of Size (2x2)
model.add(layers.MaxPooling2D((2, 2)))

#64 layers of size 3x3 and Relu Activation
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes,activation='softmax'))

model.summary()

epochs_ = 20
model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
history = model.fit(treino_gerador, batch_size=batch_size, epochs=epochs_, validation_data=validacao_gerador)

plot_history(history)

data_augmentation = tf.keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(input_shape_[0],
                                  input_shape_[1],
                                  3)),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.25),    
    layers.RandomContrast(0.05)
   
  ]
)


#Define resnet archictecture without FC (top) and without pre-trained weights
conv_layers = MobileNetV3Small(weights=None, include_top=False, input_shape=(img_width, img_height, 3))
                                                    
#Conv Layers will be tunned
conv_layers.trainable = True

model = tf.keras.Sequential([  
  conv_layers,    
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, activation='softmax')
])

model.summary()

epochs_ = 10
model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
history = model.fit(treino_gerador, batch_size= batch_size, epochs=10, validation_data=validacao_gerador)

plot_history(history)