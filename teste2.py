import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.environ["CUDNN_PATH"] = "/home/lucas/Documents/.venv/lib/python3.11/site-packages/nvidia/cudnn"
os.environ["LD_LIBRARY_PATH"] = "$CUDNN_PATH:$LD_LIBRARY_PATH"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda/"

# Substitua 'caminho_do_arquivo.csv' pelo caminho real do seu arquivo CSV
caminho_arquivo_csv = 'Datasets_csv/df_PUC.csv'
dados = pd.read_csv(caminho_arquivo_csv)

# Visualize os dados (opcional)
print(dados.head())


# Parâmetros
num_epochs = 1  # Número de épocas (passadas pelo dataset)
shuffle_buffer = 1000  # Buffer para embaralhamento

label_column = 'classe'  

# Função para carregar o dataset
def carregar_dataset(caminho_arquivo_csv, label_column='classe', batch_size=1, num_epochs=None):
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=caminho_arquivo_csv,
        batch_size=batch_size,
        label_name=label_column,
        num_epochs=num_epochs,
    )
    return dataset

# Criar o dataset
dataset = carregar_dataset(caminho_arquivo_csv, label_column)

# Verificar o número total de exemplos no dataset
num_exemplos = sum(1 for _ in dataset)

print("Número total de exemplos no dataset:", num_exemplos)


# Exemplo de divisão em treino, validação e teste
train_size = int(0.6 * dados.shape[0])  # 60% dos dados para treino
val_size = int(0.2 * dados.shape[0])    # 20% dos dados para validação
test_size = dados.shape[0] - train_size - val_size  # Restante para teste

# Função para dividir o dataset
def split_dataset(dataset, train_size, val_size, test_size):
    # Shuffle os dados
    dataset = dataset.shuffle(buffer_size=dados.shape[0], reshuffle_each_iteration=False)
    
    # Divisão dos datasets
    treino_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    teste_dataset = dataset.skip(train_size + val_size).take(test_size)
    
    return treino_dataset, val_dataset, teste_dataset

# Dividir o dataset
treino_dataset, val_dataset, teste_dataset = split_dataset(dataset, train_size, val_size, test_size)

# Verificar o número de exemplos em cada dataset (opcional)
print("Número de exemplos no conjunto de treino:", len(list(treino_dataset)))
print("Número de exemplos no conjunto de validação:", len(list(val_dataset)))
print("Número de exemplos no conjunto de teste:", len(list(teste_dataset)))

