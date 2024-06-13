import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.environ["CUDNN_PATH"] = "/home/lucas/Documents/.venv/lib/python3.11/site-packages/nvidia/cudnn"
os.environ["LD_LIBRARY_PATH"] = "$CUDNN_PATH:$LD_LIBRARY_PATH"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda/"

caminho_arquivo_csv = 'Datasets_csv/df_PUC.csv'
dados = pd.read_csv(caminho_arquivo_csv)

print(dados.head())


num_epochs = 1  
shuffle_buffer = 1000 
label_column = 'classe'  

def carregar_dataset(caminho_arquivo_csv, label_column='classe', batch_size=1, num_epochs=None):
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=caminho_arquivo_csv,
        batch_size=batch_size,
        label_name=label_column,
        num_epochs=num_epochs,
    )
    return dataset

dataset = carregar_dataset(caminho_arquivo_csv, label_column)

num_exemplos = sum(1 for _ in dataset)

print("Número total de exemplos no dataset:", num_exemplos)


train_size = int(0.6 * dados.shape[0])  # 60% 
val_size = int(0.2 * dados.shape[0])    # 20% 
test_size = dados.shape[0] - train_size - val_size

def split_dataset(dataset, train_size, val_size, test_size):
    treino_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    teste_dataset = dataset.skip(train_size + val_size).take(test_size)
    
    return treino_dataset, val_dataset, teste_dataset

treino_dataset, val_dataset, teste_dataset = split_dataset(dataset, train_size, val_size, test_size)

print("Número de exemplos no conjunto de treino:", len(list(treino_dataset)))
print("Número de exemplos no conjunto de validação:", len(list(val_dataset)))
print("Número de exemplos no conjunto de teste:", len(list(teste_dataset)))

