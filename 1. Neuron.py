# A neuron is basically a unit which take input and a wheight assostiated with each input and plus a bias

# we represnt inputs[x1,x2,x3,...,xn-1,xn] and weights[w1,w2.w3,...,wn-1,wn] as an array or list  and bias is single intger or float n


# so basically what a neuron does is it takes input muliplies each input with its corrosopnding weight 
# and add bias and return an output

inputs = [1,2,3]
weights = [0.9,0.5,-0.3]
bias = 1

ouput = (inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2])  + bias


print(ouput)




#Better way to do the opertaion above that can scale with number of inputs and weights

ouput = bias
assert len(inputs) == len(weights)
for i in range(len(inputs)):
    ouput += inputs[i] * weights[i]



print(ouput)