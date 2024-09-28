# we can use numpy libraries now because we already demonestrated that we can use our own implementaiton
import numpy
# our Layer has 3 neurons it can be extended up to n number of neurons
# the number of neurons is up to us
# the number of weights is  input x input
# the number of bias is the same as number of neurons



# 
inputs = [1.0,2.0,3.0,2.5] # 1 inputs

#
weights =[[0.2,0.8,-0.5,1],
          [0.5,-0.91,0.26,-0.5],
          [-0.26,-0.27,-0.17,-0.87]]

bias = [2.0,3.0,0.5]




# to check the output

import numpy
print(numpy.dot(weights,inputs.T))
