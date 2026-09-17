# Example 3.28c LazyPredict classification plot
# Run after Example 3.28b.py in the same notebook/session.
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(models.index, models["Accuracy"])
plt.show()
