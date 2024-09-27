# we will use previously implemented numpy library
from _3_Neurons_using_DotProduct import numpi
numpi = numpi()
# our Layer has 4 nurons it can be extended up to n number of neurons
# the number of neurons is equal to length of inputs
# the number of weights is  input x input
# the number of bias is the same as number of neurons



# 
inputs = [1.0,2.0,3.0,1.8]

#
weights =  [
        [0.1,0.2,-0.3,0.4],
        [0.7,0.9,0.4,0.6],
        [0.1,-0.4,-0.1,-0.5],
        [0.8,0.6,0.3,-0.1]
           ]
bias = [1.0,5.0,-4.0,7.0]
output = []
for i in range(len((numpi.dot(inputs,numpi.t(weights))))):
    output.append(i+bias[i])

print(output)




# to check the output

import numpy
print(numpy.dot(inputs,weights)+bias)
