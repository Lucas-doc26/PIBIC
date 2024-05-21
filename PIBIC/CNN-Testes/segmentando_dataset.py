import pandas as pd
import os
import random

locais = ['PUC', 'UFPR04', 'UFPR05']

limites = {
    'PUC': 5000,
    'UFPR04': 1000,
    'UFPR05': 1000
}

tempos = ['Cloudy', 'Rainy', 'Sunny']

for local in locais:
    caminhos_imagem = []
    classes = []
    for tempo in tempos:
        sample_dir = os.path.join(
            r'/home/lucas/Downloads/PKLot/PKLotSegmented',
            local, tempo)
        pastas = os.listdir(sample_dir)
        for pasta in pastas:
            todos_arquivos = []
            for class_dir in ['Empty', 'Occupied']:
                # Caminho completo para o subdiretório da classe
                full_class_dir = os.path.join(sample_dir, pasta, class_dir)
                if os.path.exists(full_class_dir):
                    for file in os.listdir(full_class_dir):
                        if file.endswith('.jpg'):
                            todos_arquivos.append((os.path.join(full_class_dir, file), class_dir))
                else:
                    print(f'Diretório não encontrado: {full_class_dir}')

            random.shuffle(todos_arquivos)

            # Adicionando todas as imagens ao DataFrame
            for file_path, class_dir in todos_arquivos:
                caminhos_imagem.append(file_path)
                classes.append(class_dir)

    # Embaralhando novamente para garantir a aleatoriedade
    combined_data = list(zip(caminhos_imagem, classes))
    random.shuffle(combined_data)
    caminhos_imagem, classes = zip(*combined_data)

    limite_arquivos = limites[local]

    # Limitando o número de imagens conforme o limite definido
    caminhos_imagem = caminhos_imagem[:limite_arquivos]
    classes = classes[:limite_arquivos]

    df = pd.DataFrame({
        'caminho_imagem': caminhos_imagem,
        'classe': classes
    })

    df.to_csv(f'PIBIC/CNN-estacionamento/Datasets/df_{local}.csv', index=False)

    print(f'DataFrame do local {local}:')
    print(df.head())
    print('\n')
