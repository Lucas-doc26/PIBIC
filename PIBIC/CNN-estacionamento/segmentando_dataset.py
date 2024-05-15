import pandas as pd
import os
import random

locais = ['PUC', 'UFPR04', 'UFPR05']

limites = {
    'PUC': 5000,
    'UFPR04': 1000,
    'UFPR05': 1000
}

for local in locais:
    sample_dir = os.path.join(
        r'C:\Users\lucaa\OneDrive\Área de Trabalho\pythonProject\PIBIC\CNN-estacionamento\datasets\PKLot\PKLot\PKLotSegmented',
        local, 'geral')

    caminhos_imagem = []
    classes = []
    todos_arquivos = []

    for class_dir in ['ocupada', 'vaga']:
        # Caminho completo para o subdiretório da classe
        full_class_dir = os.path.join(sample_dir, class_dir)

        # Verifique se o diretório existe antes de listar os arquivos
        if os.path.exists(full_class_dir):
            # Percorra cada arquivo de imagem no subdiretório
            for file in os.listdir(full_class_dir):
                # Verifique se o arquivo termina com '.jpg'
                if file.endswith('.jpg'):
                    todos_arquivos.append((os.path.join(full_class_dir, file), class_dir))
        else:
            print(f'Diretório não encontrado: {full_class_dir}')

    # Embaralhe a lista de arquivos de forma aleatória
    random.shuffle(todos_arquivos)

    limite_arquivos = limites[local]

    for i, (file_path, class_dir) in enumerate(todos_arquivos):
        if i < limite_arquivos:
            caminhos_imagem.append(file_path)
            classes.append(class_dir)
        else:
            # Se o limite for atingido, interrompa o loop
            break

    # Crie um DataFrame com os caminhos de imagem e classes
    df = pd.DataFrame({
        'caminho_imagem': caminhos_imagem,
        'classe': classes
    })

    df.to_csv(f'datasets/df_{local}.csv', index=False)

    print(f'DataFrame do local {local}:')
    print(df.head())
    print('\n')
