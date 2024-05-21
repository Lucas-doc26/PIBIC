from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

cancer = load_breast_cancer()

data_frame = pd.DataFrame(cancer.data, columns=list(cancer.feature_names))
data_frame['classe'] = cancer.target

classes = cancer.target_names
data_frame['Diagnostico'] = data_frame['classe'].apply(lambda x: cancer.target_names[x])
data_frame = data_frame.sample(frac=1).reset_index(drop=True)
print(data_frame)

features = data_frame.drop(['classe', 'Diagnostico'], axis=1)
labels = data_frame['classe']

x_treino, x_teste, y_treino, y_teste = train_test_split(features, labels, test_size=0.3, random_state=42)
x_treino, x_validacao, y_treino, y_validacao = train_test_split(x_treino, y_treino, train_size=0.7, test_size=0.3, random_state=42)

modelo = KNeighborsClassifier(n_neighbors=5)
modelo.fit(x_treino, y_treino)

teste_modelo = modelo.predict(x_teste)

precisao = accuracy_score(y_teste, teste_modelo)
print("A precisão do modelo é de: " + str(precisao))

contagem_classes = data_frame['classe'].value_counts()
nomes_classes = cancer.target_names

plt.figure(figsize=(8, 6))
plt.bar(nomes_classes, contagem_classes, color=['blue', 'orange'])

plt.title('Distribuição das Classes')
plt.xlabel('Classe')
plt.ylabel('Número de Amostras')
plt.show()
