# Example 3.27 PyCaret demo
from sklearn import datasets
from pycaret import classification

iris = datasets.load_iris(as_frame=True)
iris.data["Target"] = iris.target
classification.setup(data=iris.data, target="Target")
classification.compare_models()
