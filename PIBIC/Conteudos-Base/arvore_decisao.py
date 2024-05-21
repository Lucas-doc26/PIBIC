import pandas as pd
from sklearn.datasets import load_iris
data = load_iris()
iris = pd.DataFrame(data.data)
iris.columns = data.feature_names
iris['target'] = data.target
print(iris.head(3))

#filtrando somente as colunas que eu quero
iris1 = iris.loc[iris.target.isin([1,2]), ['petal length (cm)','petal width (cm)','target']]

#separando x e y
x = iris1.drop('target', axis=1)
y = iris1.target

#treino/teste
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=42)

#plotar
from matplotlib import pyplot as plt

fig, ax = plt.subplots()
ax.scatter(x_treino['petal length (cm)'],
           x_treino['petal width (cm)'],
           c=y_treino)
plt.show()

#arvore de decisão
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_treino, y_treino)
print("O score:", clf.score(x_treino, y_treino))

#regras da árvore de treino
fig, ax = plt.subplots()
tree.plot_tree(clf)
plt.show()

#decisão da árvore 1
fig, ax = plt.subplots()
ax.scatter(x_treino['petal length (cm)'],
           x_treino['petal width (cm)'],
           c=y_treino)
ax.set(xlim=(2.9, 7), ylim=(0.9, 2.7))
ax.plot([5.05,5.05], [0.9,2.7], '--r')
plt.show()

#decisão da árvore 2
fig, ax = plt.subplots()
ax.scatter(x_treino['petal length (cm)'],
           x_treino['petal width (cm)'],
           c=y_treino)
ax.set(xlim=(2.9, 7), ylim=(0.9, 2.7))
ax.plot([5.05,5.05], [0.9,2.7], '--r')
ax.plot([2.9,5.05], [1.9,1.9], '--r')
plt.show()

#decisão da árvore 3
fig, ax = plt.subplots()
ax.scatter(x_treino['petal length (cm)'],
           x_treino['petal width (cm)'],
           c=y_treino)
ax.set(xlim=(2.9, 7), ylim=(0.9, 2.7))
ax.plot([5.05,5.05], [0.9,2.7], '--r')
ax.plot([2.9,5.05], [1.9,1.9], '--r')
ax.plot([2.9,5.05], [1.65,1.65], '--r')
plt.show()

#decisão da árvore 4
fig, ax = plt.subplots()
ax.scatter(x_treino['petal length (cm)'],
           x_treino['petal width (cm)'],
           c=y_treino)
ax.set(xlim=(2.9, 7), ylim=(0.9, 2.7))
ax.plot([5.05,5.05], [0.9,2.7], '--r')
ax.plot([2.9,5.05], [1.9,1.9], '--r')
ax.plot([2.9,5.05], [1.65,1.65], '--r')
ax.plot([4.65,4.65], [1.65,1.9], '--r')
plt.show()

#previsão e avaliando o erro
y_pred = clf.predict(x_teste)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_teste, y_pred))

fig, ax = plt.subplots()
ax.scatter(x_teste['petal length (cm)'],
           x_teste['petal width (cm)'],
           c=y_teste)
ax.set(xlim=(2.9, 7), ylim=(0.9, 2.7))
ax.plot([5.05,5.05], [0.9,2.7], '--r')
ax.plot([2.9,5.05], [1.9,1.9], '--r')
ax.plot([2.9,5.05], [1.65,1.65], '--r')
ax.plot([4.65,4.65], [1.65,1.9], '--r')
plt.show()


#Agora fazendo com a biblioteca inteira
x = iris.drop('target', axis=1)
y = iris['target']

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=42)
clf2 = tree.DecisionTreeClassifier(random_state=42).fit(x_treino, y_treino)
print("O score:", clf2.score(x_treino, y_treino))

fig, ax = plt.subplots(figsize=(10,8))

tree.plot_tree(clf2)
plt.show()

y_pred2 = clf2.predict(x_teste)
print(confusion_matrix(y_teste, y_pred2))
