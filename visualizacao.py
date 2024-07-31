import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def plot_history(history):
  """
  Irá plotar o seu history(modelo treinado), fazendo dois gráficos: \n
  época X accuracy \n
  época X loss
  """

  print(history.history.keys())
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
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

def plot_imagens_com_csv(dataframe, img_por_coluna):
    """
    Função para plotar imagens do DataFrame pegando o .Csv.
    :param dataframe: DataFrame contendo caminhos de imagens .csv
    :param img_por_coluna: número de imagens por coluna
    """
    _, axs = plt.subplots(img_por_coluna, img_por_coluna, figsize=(10, 10))

    # Iterar sobre as linhas do DataFrame
    for index, linha in dataframe.iterrows():
        if index >= img_por_coluna**2:
            break  
        image_path = linha['caminho_imagem']
        image = plt.imread(image_path)
        linha_idx = index // img_por_coluna
        coluna_idx = index % img_por_coluna
        axs[linha_idx, coluna_idx].imshow(image)
        axs[linha_idx, coluna_idx].axis('off')  # Desligar os eixos
        axs[linha_idx, coluna_idx].set_title(f"{linha['classe']}")  # Título da imagem

    plt.tight_layout()
    plt.show()
     
def plot_imagens_dataframe_gerador(dataframe_gerador, img_por_coluna):
    """
    Função para plotar imagens geradas pelo DataFrameIterator.
    :param dataframe_gerador: DataFrameIterator contendo as imagens geradas.
    :param img_por_coluna: Número de imagens por coluna e por linha a serem plotadas.
    """
    fig, axs = plt.subplots(img_por_coluna, img_por_coluna, figsize=(10, 10))
    imagens_por_plot = img_por_coluna ** 2 
    
    for i in range(imagens_por_plot):
        if i >= len(dataframe_gerador.filenames):
            break
        
        # Carrega um batch de imagens e labels
        batch = next(dataframe_gerador)
        imagens = batch[0]
        labels = batch[1]
        
        # Itera sobre as imagens no batch
        for j in range(len(imagens)):
            image = imagens[j]
            label = labels[j]
            
            # Normaliza os valores dos pixels para o intervalo [0, 1]
            image = (image - image.min()) / (image.max() - image.min())
            
            # Plota a imagem no subplot correspondente
            axs[i // img_por_coluna, i % img_por_coluna].imshow(image)
            axs[i // img_por_coluna, i % img_por_coluna].axis('off')  
            axs[i // img_por_coluna, i % img_por_coluna].set_title(f"Classe: {label}")  # Título da imagem

    plt.tight_layout()
    plt.show()

def show_sample(dataset):
    dataset = dataset.take(9)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        image, label = next(iter(dataset))
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image.numpy().astype("uint8"))
        plt.title(label.numpy().astype("uint8"))
        plt.axis("off")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels, save_path, title):
    """
    Plota uma matriz de confusão.

    Args:
    - y_true (array): rótulos verdadeiros.
    - y_pred (array): rótulos previstos.
    - labels (list): lista de rótulos das classes.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()

def exibir_primeiras_imagens(dataframe):
    # Pegar os caminhos das 9 primeiras imagens
    caminhos_imagens = dataframe['caminho_imagem'].iloc[:9].tolist()

    # Configurar a exibição das imagens
    plt.figure(figsize=(12, 8))
    for i, caminho in enumerate(caminhos_imagens):
        plt.subplot(3, 3, i + 1)
        img = Image.open(caminho)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Imagem {i + 1}')
    plt.tight_layout()
    plt.show()

def plot_imagens_incorretas(y_binario, y_predicao, caminhos_imagens, modelo_nome:str, dataset_nome:str, n_imagens_por_grade:3):
    """
    Plota imagens cujas previsões foram incorretas.

    Parâmetros:
    - y_binario: Array numpy com os rótulos reais.
    - y_predicao: Array numpy com as previsões do modelo.
    - caminhos_imagens: Lista de caminhos para as imagens correspondentes.
    - modelo_nome: Nome do modelo para fins de salvamento de arquivo.
    - dataset_nome: Nome do dataset para fins de salvamento de arquivo.
    - n_imagens_por_grade: Número de imagens por grade na visualização (default: 3).

    A função identifica as imagens onde as previsões do modelo diferem dos rótulos reais,
    e plota estas imagens junto com suas previsões e rótulos reais. As imagens são organizadas
    em uma grade de subplots para visualização fácil. As imagens incorretamente previstas são
    salvas em um arquivo PNG com base nos nomes do modelo e do dataset fornecidos.
    """

    labels = ['Empty', 'Occupied']

    indices_incorretos = np.where(y_predicao != y_binario)[0]

    num_imagens_plotadas = min(len(indices_incorretos), n_imagens_por_grade**2)
    indices_plotados = indices_incorretos[:num_imagens_plotadas]

    fig, axes = plt.subplots(n_imagens_por_grade, n_imagens_por_grade, figsize=(15, 15))
    axes = axes.flatten()

    plt.suptitle(f'{modelo_nome} vs {dataset_nome} - Imagens incorretas', fontsize=16)
    for ax, indice in zip(axes, indices_plotados):
        img = load_img(caminhos_imagens[indice])
        ax.imshow(img)
        ax.set_title(f'Predição: {labels[y_predicao[indice]]}\nReal: {labels[y_binario[indice]]}')
        ax.axis('off')

    # Remover qualquer eixo vazio
    for i in range(num_imagens_plotadas, n_imagens_por_grade**2):
        axes[i].axis('off')

    plt.tight_layout()
    save_path_imgs = os.path.join('Resultados', 'Imagens_Incorretas', modelo_nome, f'{modelo_nome}_vs_{dataset_nome}_imagens_incorretas.png')
    os.makedirs(os.path.dirname(save_path_imgs), exist_ok=True)
    plt.savefig(save_path_imgs)
    plt.close()

