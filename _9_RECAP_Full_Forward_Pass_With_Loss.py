import os, psutil

import nnfs 
import nnfs.datasets
import numpy as np




class DenseLayer:
    def __init__(self,number_of_features,number_of_neurons) -> None:  # inititation of weights and bias
        self.weights =  0.01 * np.random.randn(number_of_features,number_of_neurons)
        self.bias = np.zeros((1,number_of_neurons))

    def forward(self,inputs):  # sets self.output to dot product
        self.output = np.dot(inputs,self.weights) + self.bias

    def samples(self,inputs): # sets self.output to the input data
        self.output = inputs




class Activation:
    def Relu_forward(self,inputs):
        self.output = np.maximum(0,inputs)
    def SoftMax_forward(self,inputs):
        exponent_values = np.exp(inputs - np.max(inputs , axis=1 , keepdims=True))
        self.output = exponent_values/np.sum(exponent_values,axis=1,keepdims=True)



class Loss:
    # Categorical Crossentropy Loss Calculation
    def Loss_CategoricalCrossentropy(self, Predicted_Value, Expected_TrueValue):
        # Clip predicted values to prevent log(0)
        pred_clipped = np.clip(Predicted_Value, 1e-7, 1 - 1e-7)

        # If Expected_TrueValue is a 1D array 
        if len(Expected_TrueValue.shape) == 1:
            # Select the predicted confidence values for the correct class
            # It selects from row Number (range(len(pre_clipped)))  to column (expected True Values) Eg row 0 column 1  then row 0 column 2
            correct_confidences = pred_clipped[range(len(pred_clipped)), Expected_TrueValue]   
            
        

        # If Expected_TrueValue is a 2D array (one-hot encoded labels)
        elif len(Expected_TrueValue.shape) == 2:
            # Calculate the correct confidences for one-hot encoded labels
            correct_confidences = np.sum(pred_clipped * Expected_TrueValue, axis=1)
            
        

        # Calculate the loss
        self.CategoricalCrossentropy_loss = -np.log(correct_confidences)
        
        return self.CategoricalCrossentropy_loss
    
    # Mean Loss Calculation
    def Mean_Loss(self):
        # Return the mean of all individual losses

        return np.mean(self.CategoricalCrossentropy_loss)
    


    
# creating dataset
nnfs.init()
X,Y = nnfs.datasets.spiral_data(samples=100,classes=3) # Create spiral data set for training

# we are going to have 1 input layer(two neurons), 2 hidden layer(4 neruons each) and 1 output layer(with 2 output neuron)
loss = Loss() 

InputLayer = DenseLayer(2,2)



HiddenLayer1 = DenseLayer(2,4)
HiddenDenseLayer1Activation = Activation()

HiddenLayer2 = DenseLayer(4,4)
HiddenDenseLayer2Activation = Activation()

OutputLayer = DenseLayer(4,3)
OutputDenseLayerActivaiton = Activation()

# starting forward pass

InputLayer.samples(X)

HiddenLayer1.forward(InputLayer.output)
HiddenDenseLayer1Activation.Relu_forward(HiddenLayer1.output)


HiddenLayer2.forward(HiddenDenseLayer1Activation.output)
HiddenDenseLayer2Activation.Relu_forward(HiddenLayer2.output)

OutputLayer.forward(HiddenDenseLayer2Activation.output)

OutputLayer.output = np.array([[0.5586645,0.4660373,0.4161023]])

OutputDenseLayerActivaiton.SoftMax_forward(OutputLayer.output)





loss_Value = loss.Loss_CategoricalCrossentropy(OutputDenseLayerActivaiton.output,Y)
AverageLoss = loss.Mean_Loss()


#argmax return 
accuracy = np.mean( np.argmax(OutputDenseLayerActivaiton.output, axis=1) == Y)



print("Loss : {} ".format(AverageLoss))
print("Accuracy : {}% ".format((accuracy*100).round(3)))





print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB")
