import pandas as pd
import os
import random
import csv
from typing import Tuple, Optional

def segmentadando_datasets(quantidade_PUC:int=None, quantidade_UFPR04:int=None, quantidade_UFPR05:int=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Função para criar os datasets, retorna 3 DataFrames separados, um para cada faculdade.
    """
    faculdades = ['PUC', 'UFPR04', 'UFPR05']
    
    limites_padrao = {
        'PUC': float('inf'),  # Infinito para capturar todas as imagens
        'UFPR04': float('inf'),
        'UFPR05': float('inf')
    }
    
    if quantidade_PUC is not None:
        limites_padrao['PUC'] = quantidade_PUC
    if quantidade_UFPR04 is not None:
        limites_padrao['UFPR04'] = quantidade_UFPR04
    if quantidade_UFPR05 is not None:
        limites_padrao['UFPR05'] = quantidade_UFPR05

    tempos = ['Cloudy', 'Rainy', 'Sunny']
    
    dataframes = [] 

    for local in faculdades:
        caminhos_imagem = []
        classes = []
        for tempo in tempos:
            sample_dir = os.path.join(
                r"/home/lucas/Downloads/PKLot/PKLotSegmented/",
                local, tempo)
                #"C:\Users\lucaa\Downloads\PKLot\PKLot\PKLotSegmented"
                #/home/lucas/Downloads/PKLot/PKLotSegmented/
            pastas = os.listdir(sample_dir)
            
            if not os.path.exists(sample_dir)   :
                print(f'Diretório não encontrado: {sample_dir}')

            for pasta in pastas:
                todos_arquivos = []
                for class_dir in ['Empty', 'Occupied']:
                    # Caminho completo para o subdiretório da classe
                    full_class_dir = os.path.join(sample_dir, pasta, class_dir)
                    if os.path.exists(full_class_dir):
                        for file in os.listdir(full_class_dir):
                            if file.endswith('.jpg'):
                                todos_arquivos.append((os.path.join(full_class_dir, file), class_dir))

                random.shuffle(todos_arquivos)

                # Adicionando todas as imagens ao DataFrame
                for file_path, class_dir in todos_arquivos:
                    caminhos_imagem.append(file_path)
                    classes.append(class_dir)

        # Embaralhando novamente para garantir a aleatoriedade
        combined_data = list(zip(caminhos_imagem, classes))
        random.shuffle(combined_data)
        caminhos_imagem, classes = zip(*combined_data)

        limite_arquivos = min(limites_padrao[local], len(caminhos_imagem))

        caminhos_imagem = caminhos_imagem[:limite_arquivos]
        classes = classes[:limite_arquivos]

        df = pd.DataFrame({
            'caminho_imagem': caminhos_imagem,
            'classe': classes
        })

        # Salvar o DataFrame como arquivo CSV
        csv_path = f'Datasets_csv/df_{local}.csv'
        df.to_csv(csv_path, index=False)
        print(f'DataFrame do local {local} salvo como: {csv_path}')

        # Adicionando o DataFrame à lista
        dataframes.append(df)

        print(f'DataFrame do local {local}:')
        print(df.head())
        print('\n')

    # Retornar os DataFrames separados
    return tuple(dataframes)

# Exemplo de usos: 
#dataframePuc1, dataframeUFPR041, dataframeUFPR051 = segmentadando_datasets()
dataframePuc, dataframeUFPR04, dataframeUFPR05 = segmentadando_datasets(1000, 1000, 1000)

    
def csv_para_dicionario(caminho):
    # Dicionário para armazenar os dados do CSV
    dados = {}

    # Abre o arquivo CSV
    with open(caminho, mode='r', newline='') as arquivo_csv:
        # Cria um leitor CSV
        leitor_csv = csv.DictReader(arquivo_csv)
        
        # Itera sobre as linhas do arquivo CSV
        for linha in leitor_csv:
            # Adiciona os dados da linha ao dicionário
            caminho_imagem = linha['caminho_imagem']
            classe = linha['classe']
            dados[caminho_imagem] = classe

    return dados

# Exemplo de uso da função
"""dicionario = csv_para_dicionario('Datasets_csv\df_PUC.csv')
print(dicionario)
print(dicionario[1])"""

"""df = pd.read_csv('Datasets_csv/df_PUC.csv')
print(df.head())"""