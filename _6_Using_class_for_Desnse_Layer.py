import nnfs
import nnfs.datasets
import numpy as np



class DenseLayer:
    def __init__(self,number_of_features,number_of_neurons) -> None:
        self.weights =  0.01 * np.random.randn(number_of_features,number_of_neurons)
        self.bias = np.zeros((1,number_of_neurons))

    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.bias




nnfs.init()
x,y = nnfs.datasets.spiral_data(samples=100,classes=3)
print(x,end='\n\n')

dense1 = DenseLayer(2,3)

dense1.forward(x)
print(dense1.output)