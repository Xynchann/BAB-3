# Example 3.28e LazyPredict regression plot
# Run after Example 3.28d.py in the same notebook/session.
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(models.index, models["R-Squared"], "-s")
plt.show()
