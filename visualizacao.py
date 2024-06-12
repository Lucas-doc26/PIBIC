import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas

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

    for index, linha in dataframe.take(9):
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
    Função para plotar imagens geradas pelo DataFrameGenerator.
    :param dataframe_gerador: DataFrameIterator contendo as imagens geradas.
    """
    _, axs = plt.subplots(img_por_coluna, img_por_coluna, figsize=(10, 10))
    imagens = img_por_coluna ** 2 

    for i in range(imagens):
        if i >= imagens:
            break  
        batch = next(dataframe_gerador)
        image = batch[0][0]  
        label = batch[1][0] 
        
        # Normaliza os valores dos pixels para o intervalo [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        
        axs[i // 3, i % 3].imshow(image)
        axs[i // 3, i % 3].axis('off')  
        axs[i // 3, i % 3].set_title(f"Classe: {label}")  # Título da imagem

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

def plot_confusion_matrix2(y_true, y_pred, labels):
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
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.close()