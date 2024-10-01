import numpy as np 
import nnfs
import nnfs.datasets


nnfs.init() # setup to intialize to get consistent output in numpy
X,Y = nnfs.datasets.spiral_data(samples=100,classes=3) # Create spiral data set for training



class DenseLayer:
    def __init__(self,numberofFeatures,numberOfNeurons) -> None:
        self.weight = 0.1* np.random.rand(numberofFeatures,numberOfNeurons)
        self.bias = np.zeros((1,numberOfNeurons))
        

    def forward(self,inputs):
        self.output = np.dot(inputs,self.weight) + self.bias



class Activation:
    def Relu(self,inputs):
        self.output = np.maximum(0,inputs)


InputDenseLayer = DenseLayer(2,4)
InputDenseLayerActivaiton = Activation()

FirstHiddenDenseLayer = DenseLayer(4,4)
FirstHiddenDenseLayerActivation = Activation()

SecondHiddenDenseLayer = DenseLayer(4,4)
SecondHiddenDenseLayerActivation = Activation()



OutputDenseLayer = DenseLayer(4,2)




InputDenseLayer.forward(X)
InputDenseLayerActivaiton.Relu(InputDenseLayer.output)



FirstHiddenDenseLayer.forward(InputDenseLayerActivaiton.output)
FirstHiddenDenseLayerActivation.Relu(FirstHiddenDenseLayer.output)

SecondHiddenDenseLayer.forward(FirstHiddenDenseLayerActivation.output)
SecondHiddenDenseLayerActivation.Relu(SecondHiddenDenseLayer.output)


OutputDenseLayer.forward(SecondHiddenDenseLayerActivation.output)



