import numpy as np 
import nnfs
import nnfs.datasets


nnfs.init() # setup to intialize to get consistent output in numpy
X,Y = nnfs.datasets.spiral_data(samples=100,classes=3) # Create spiral data set for training

print(Y)

class DenseLayer:
    def __init__(self,numberofFeatures,numberOfNeurons) -> None:
        self.weight = 0.1* np.random.rand(numberofFeatures,numberOfNeurons)
        self.bias = np.zeros((1,numberOfNeurons))
        

    def forward(self,inputs):
        self.output = np.dot(inputs,self.weight) + self.bias



class Activation:
    def Relu_forward(self,inputs):
        self.output = np.maximum(0,inputs)
    def SoftMax_forward(self,inputs):
        exponent_values = np.exp(inputs - np.max(inputs , axis=1 , keepdims=True))
        self.output = exponent_values/np.sum(exponent_values,axis=1,keepdims=True)



InputDenseLayer = DenseLayer(2,4)
InputDenseLayerActivaiton = Activation()

FirstHiddenDenseLayer = DenseLayer(4,4)
FirstHiddenDenseLayerActivation = Activation()

SecondHiddenDenseLayer = DenseLayer(4,4)
SecondHiddenDenseLayerActivation = Activation()



OutputDenseLayer = DenseLayer(4,3)
OutputDenseLayerActivation = Activation()





InputDenseLayer.forward(X)
InputDenseLayerActivaiton.Relu_forward(InputDenseLayer.output)



FirstHiddenDenseLayer.forward(InputDenseLayerActivaiton.output)
FirstHiddenDenseLayerActivation.Relu_forward(FirstHiddenDenseLayer.output)

SecondHiddenDenseLayer.forward(FirstHiddenDenseLayerActivation.output)
SecondHiddenDenseLayerActivation.Relu_forward(SecondHiddenDenseLayer.output)


OutputDenseLayer.forward(SecondHiddenDenseLayerActivation.output)
OutputDenseLayerActivation.SoftMax_forward(OutputDenseLayer.output)



print(OutputDenseLayerActivation.output)



