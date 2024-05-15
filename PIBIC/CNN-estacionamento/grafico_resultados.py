import matplotlib.pyplot as plt

modelos_testes = [
    "Dropout(25%, 25%, 50%) + Kernel regularizer",
    "Dropout(25%, 25%) + Kernel regularizer",
    "Dropout(25%) + Kernel regularizer",
    "Kernel regularizer",
    "Dropout(25%, 25%, 50%)",
    "Dropout(25%, 25%)",
    "Dropout(25%)",
    "Sem ajustes de Dropout e Kernel"
]

# Accuracy
accuracies = [
    [0.999, 0.932, 0.62],
    [0.997, 0.931, 0.81],
    [0.997, 0.973, 0.673],
    [0.998, 0.96, 0.703],
    [0.998, 0.972, 0.759],
    [0.999, 0.939, 0.854],
    [0.994, 0.881, 0.711],
    [0.996, 0.948, 0.581]
]

# Organizando os dados em listas separadas
num_modelos_testes = len(modelos_testes)
num_acuracias = len(accuracies[0])
num_barras = len(modelos_testes)

largura_barra = 0.2
posicoes = range(1, num_barras + 1)

nome_modelo = {
    0: "PUCPR",
    1: "UFPR04",
    2: "UFPR05"
}

plt.figure(figsize=(12, 8))
for i in range(num_acuracias):
    posicoes_deslocadas = [x + i * largura_barra for x in posicoes]
    plt.bar(posicoes_deslocadas, [accuracies[j][i] for j in range(num_modelos_testes)], width=largura_barra, label=f'{nome_modelo[i]}')
    for x, y in zip(posicoes_deslocadas, [accuracies[j][i] for j in range(num_modelos_testes)]):
        plt.text(x, y, f'{y:.3f}', ha='center', va='bottom')
plt.xlabel('Ajustes')
plt.ylabel('Accuracy')
plt.title('Comparação de accuracy entre os ajustes')
plt.xticks(posicoes, modelos_testes, rotation=45, ha='right')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
