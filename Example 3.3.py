# Example 3.3 Python SVM Iris Classifications using sklearn dataset
from sklearn import svm
from sklearn.datasets import load_iris

# Memuat dataset iris langsung dari sklearn
iris = load_iris()

# X untuk fitur (mengambil 2 kolom pertama)
X = iris.data[:, :2] 

# y untuk label kelas bunganya
y = iris.target

# Membuat dan melatih model SVM
clf = svm.SVC()
clf.fit(X, y)

# Prediksi bunga untuk sepal length dan width tertentu
p = clf.predict([[5.4, 3.2]])
print(p)