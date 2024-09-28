import numpy as np

# Inputs and weights
inputs = np.array([[1.03, 2.26, 1.79 , 3.64],
                   [4.11, 1.92, 4.47, 2.92]])

weights1 = np.array([[3.86, 4.47, 4.85, 4.87],
                    [0.69, 0.08, 0.6 , 0.1 ],
                    [0.23, 0.28, 0.73, 0.38]])

weights2 = np.array([[0.97, 0.8 , 0.41 ],
                    [0.64, 0.41, 0.05],
                    [0.06, 0.34, 0.4 ]])

# Bias vectors
bias1 = np.array([1, 1, 1])   # 3 elements matching hidden layer size
bias2 = np.array([1, 2, 3])  # 4 elements matching output layer size

# Forward pass - Layer 1
inputLayer = np.dot(inputs, weights1.T) + bias1  # Apply weights and add bias


# Forward pass - Layer 2
outputLayer = np.dot(inputLayer, weights2.T) + bias2  # Apply second layer weights and add bias
print("Output Layer (before activation):\n", outputLayer)
