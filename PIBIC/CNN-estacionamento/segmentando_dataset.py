import pandas as pd
import os
import random

# Caminho do diretório de amostra
#sample_dir = r'C:\Users\lucaa\OneDrive\Área de Trabalho\pythonProject\PIBIC\CNN-estacionamento\datasets\PKLot\PKLot\PKLotSegmented\PUC\conjunto_5k'
#sample_dir = r'C:\Users\lucaa\OneDrive\Área de Trabalho\pythonProject\PIBIC\CNN-estacionamento\datasets\PKLot\PKLot\PKLotSegmented\UFPR04\Geral'
sample_dir = r'C:\Users\lucaa\OneDrive\Área de Trabalho\pythonProject\PIBIC\CNN-estacionamento\datasets\PKLot\PKLot\PKLotSegmented\UFPR05\Geral'
caminhos_imagem = []
classes = []

# Limite de arquivos a processar
limite_arquivos = 1000

# Lista para armazenar todos os arquivos de ambas as pastas
todos_arquivos = []

# Percorra as classes 'ocupada' e 'vaga'
for class_dir in ['ocupada', 'vaga']:
    # Caminho completo para o subdiretório da classe
    full_class_dir = os.path.join(sample_dir, class_dir)

    # Verifique se o diretório existe antes de listar os arquivos
    if os.path.exists(full_class_dir):
        # Percorra cada arquivo de imagem no subdiretório
        for file in os.listdir(full_class_dir):
            # Verifique se o arquivo termina com '.jpg'
            if file.endswith('.jpg'):
                # Adicione o caminho da imagem e a classe correspondente à lista de todos os arquivos
                todos_arquivos.append((os.path.join(full_class_dir, file), class_dir))
    else:
        print(f'Diretório não encontrado: {full_class_dir}')

# Embaralhe a lista de arquivos de forma aleatória
random.shuffle(todos_arquivos)

# Percorra a lista embaralhada e adicione os arquivos às listas 'caminhos_imagem' e 'classes'
for i, (file_path, class_dir) in enumerate(todos_arquivos):
    if i < limite_arquivos:
        # Adicione o caminho da imagem à lista
        caminhos_imagem.append(file_path)

        # Adicione a classe correspondente à lista
        if class_dir == 'ocupada':
            classes.append('ocupada')
        elif class_dir == 'vaga':
            classes.append('vaga')
    else:
        # Se o limite for atingido, interrompa o loop
        break

# Crie um DataFrame com os caminhos de imagem e classes
df = pd.DataFrame({
    'caminho_imagem': caminhos_imagem,
    'classe': classes
})

# Exiba as primeiras linhas do DataFrame
print(df.head())
df.to_csv('datasets\df_ufpr05.csv', index=False)
