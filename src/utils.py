import sys
import numpy as np
import matplotlib.pyplot as plt

# Print information about our environment
print(f"Python version: {sys.version}")
print(f"Python location: {sys.executable}")
print(f"NumPy version: {np.__version__}")

# Create a simple plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Test Plot")
plt.show()