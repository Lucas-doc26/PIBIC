from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Dados fictícios: rótulos reais e previsões
y_true = ['empty', 'occupied', 'empty', 'empty', 'occupied', 'occupied', 'empty', 'occupied', 'occupied', 'empty']
y_pred = ['empty', 'occupied', 'empty', 'occupied', 'occupied', 'empty', 'empty', 'occupied', 'empty', 'occupied']

# Definir as classes
classes = ['empty', 'occupied']

# Calcular matriz de confusão
cm = confusion_matrix(y_true, y_pred, labels=classes)

# Exibir a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, values_format='.0f')
plt.title('Matriz de Confusão')
plt.xlabel('Predições')
plt.ylabel('Rótulos Reais')
plt.grid(False)
plt.show()
