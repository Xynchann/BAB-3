# Exercise 3.19: PyCaret classification using breast cancer data
# Requires PyCaret, which needs a Python version older than 3.14.
from pycaret import classification
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer(as_frame=True)
data = cancer.data.copy()
data["Target"] = cancer.target
classification.setup(data=data, target="Target")
classification.compare_models()
