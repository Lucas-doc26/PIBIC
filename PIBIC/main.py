from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

# Carregar a base de dados Wine e embaralha
wine = load_wine()

#Criando o dataframe
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
print(df.head())

#Gerar um target name
wine.target_names
df['target_names'] = wine.target_names[df['target']]
df = df.sample(frac=1).reset_index(drop=True)

#Todas as features disponiveis: ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
#Gerar as features que eu quero
wine_features = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']


data = df[wine_features]
labels = df.target

# Dividindo dados de treinamento (70%) e teste (30%)
x_treino, x_teste, y_treino, y_teste = train_test_split(data, labels, test_size=0.3, random_state=42)

# Dividindo dados de treinamento em treinamento (70%) e validação (30%)
x_treino, x_val, y_treino, y_val = train_test_split(x_treino, y_treino, train_size=0.7, test_size=0.3, random_state=42)

# Treinando o modelo
modelo = KNeighborsClassifier(3)
modelo.fit(x_treino, y_treino)

# Testando o modelo
teste_modelo = modelo.predict(x_teste)

precisao = accuracy_score(y_teste, teste_modelo)
print("Precisão do modelo:", precisao)

contagem_classes = df['target'].value_counts()
nomes_classes = wine.target_names

plt.figure(figsize=(8, 6))
plt.bar(nomes_classes, contagem_classes, color=['blue', 'orange', 'red'])

plt.title('Distribuição das Classes')
plt.xlabel('Classes de vinho')
plt.ylabel('Número de Amostras')
plt.show()