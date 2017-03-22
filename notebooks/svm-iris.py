from sklearn import datasets
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
import matplotlib.pyplot as plt
import numpy as np

# Test SVC with RBF kernel
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

h_class = SVC(C=1.0, kernel='rbf', gamma=0.7, random_state=101)

scores = cross_val_score(h_class, X, y, cv=20, scoring='accuracy')
print("Accuracy = {}".format(np.mean(scores)))

h_class.fit(X, y)
h_class.support_


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

Z = h_class.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X0, X1, c=y, s=100, edgecolor='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')


# Test SVM regressor SVR with the Boston dataset
scaler = StandardScaler()
boston = datasets.load_boston()

shuffled = np.random.permutation(boston.target.size)
X = scaler.fit_transform(boston.data[shuffled, :])
y = boston.target[shuffled]

h_regr = SVR(kernel='rbf', C=20.0, gamma=0.001, epsilon=1.0)
scores = cross_val_score(h_regr, X, y, cv=20, scoring='neg_mean_squared_error')
print("MSE = {:.3f}".format(abs(np.mean(scores))))
