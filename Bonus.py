from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data[:, :2]
y = iris.target


knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X, y)

h = .02  
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.5, cmap='Pastel2')  


scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Set1')  
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend(*scatter.legend_elements(), loc="lower right", title="Classes")
plt.title("7-Nearest Neighbors Decision Boundaries on Iris Dataset")
plt.colorbar(ticks=[0, 1, 2], label='Class')
plt.show()
