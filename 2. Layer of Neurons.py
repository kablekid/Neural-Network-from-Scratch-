inputs = [1,2,3,4]

weights1 = [0.4,0.9,0.1,0.7]
weights2 = [-0.8,-0.7,0.3,0.6]
wightts3 = [0.5,0.9,-0.5,0.8]

bias1 = 2
bias2 = -1
bias3 = 0.7



output1 = inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + inputs[3] * weights1[3] + bias1
output2 = inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + inputs[3] * weights2[3] + bias2
output3 = inputs[0] * wightts3[0] + inputs[1] * wightts3[1] + inputs[2] * wightts3[2] + inputs[3] * wightts3[3] + bias3

output = [output1,output2,output3]

print(output)


# better code to do multi later Neruons of any size



inputs = [1,2,3,4]

weights = [[0.4,0.9,0.1,0.7],
            [-0.8,-0.7,0.3,0.6],
            [0.5,0.9,-0.5,0.8]]

bias =[ 2 ,-1 ,0.7]
outputs = []

for j, weight in enumerate(weights):
    output = 0
    for i in range(len(weight)):
       output += inputs[i] * weight[i]
    output += bias[j]
    outputs.append(output)    
    
print(outputs)


# if you take a closer look we are just doing a dot product or input and weight matrix!