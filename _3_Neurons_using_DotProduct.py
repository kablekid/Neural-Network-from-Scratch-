# we said  the output of the neuron is (inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2])  + bias
# this is just the dot prouct of inputs and weights  
# we can use numpy libarry to do .dot() product but we are goona implment a class that basically do the same thing but less efficent
from lib.numpi import numpi

    
# Usage Example :

# input is a vector single batch of data
inputs = [1.0,2.0,3.0,2.5]

# weights is a matrix where each row corrosponds to each neurons 
weights = [[0.2,0.8,-0.5,1.0],
          [1.0,2.0,3.0,2.5],
          [1.0,2.0,3.0,2.5],
          [1.0,2.0,3.0,2.5]
          ]
bias = [2.0,1.36,3.9,4.4]



calculation = numpi.dot(inputs,weights) 


outputs = numpi.addBias(calculation,bias)

print(outputs)

