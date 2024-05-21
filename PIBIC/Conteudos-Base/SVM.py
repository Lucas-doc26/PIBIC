import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

data = load_iris()
x,y = load_iris(return_X_y=True, as_frame=True)

x = x.loc[y.isin([0,1]), ['petal length (cm)', 'petal width (cm)']]
y = y.loc[y.isin([0,1])]

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=42)

fig, ax = plt.subplots()
ax.scatter(x_treino['petal length (cm)'],
           x_treino['petal width (cm)'],
           c=y_treino)
plt.show()

clf = SVC(kernel='linear').fit(x_treino, y_treino)

w1 = clf.coef_[0][0]
w2 = clf.coef_[0][1]
w0 = clf.intercept_[0]

fig, ax = plt.subplots()
ax.scatter(x_treino['petal length (cm)'],
           x_treino['petal width (cm)'],
           c=y_treino)

#equação da reta(y)
x = np.linspace(1,5,100)
y = (-w1*x-w0)/w2

ax.plot(x,y,color='r')
ax.set(ylim=(0,1.8))
plt.show()

print(clf.support_vectors_)
vetor_suporte1 = clf.support_vectors_[:,0]
vetor_suporte2 = clf.support_vectors_[:,1]

#Traçando o gráfico com os vetores de suporte marcados
fig, ax = plt.subplots()
ax.scatter(x_treino['petal length (cm)'],
           x_treino['petal width (cm)'],
           c=y_treino)
ax.scatter(vetor_suporte1,vetor_suporte2,c='r')
x = np.linspace(1,5,100)
y = (-w1*x-w0)/w2

ax.plot(x,y,color='r')
ax.set(ylim=(0,1.8))
plt.show()

#Traçando o gráfico com os vetores de suporte marcados e suas linhas
fig, ax = plt.subplots()
ax.scatter(x_treino['petal length (cm)'],
           x_treino['petal width (cm)'],
           c=y_treino)
ax.scatter(vetor_suporte1,vetor_suporte2,c='r')
x = np.linspace(1,5,100)
y = (-w1*x-w0)/w2
ax.plot(x,y,'r')

y2 = (+1-w1*x-w0)/w2
ax.plot(x,y2,'--r')

y3 = (-1-w1*x-w0)/w2
ax.plot(x,y3,'--r')

ax.set(ylim=(0,1.8))
plt.show()

#Com todas as colunas:
x,y = load_iris(return_X_y=True, as_frame=True)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=42)

clf2 = SVC().fit(x_treino, y_treino)

#Avaliando o modelo:
y_pred = clf2.predict(x_teste)
print(confusion_matrix(y_teste, y_pred))
