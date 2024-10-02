import numpy as np
# Cross-entropy loss is a commonly used loss function for classification tasks in machine learning,
# particularly when dealing with problems that involve predicting a class label from multiple possible classes
# (such as in neural networks used for image classification).

# For binary classification, cross-entropy loss can be expressed as:

# 𝐿 = −[𝑦⋅log⁡(𝑝)+(1−𝑦)⋅log⁡(1−𝑝)]
#Where:  y: the true label (0 or 1)
#        p: the predicted probability for the true class (output from the model, a value between 0 and 1)

#For multi-class classification, cross-entropy loss generalizes to:

# L=− (from i to n)∑yi*log(pi)
#where:
#yi the true label for class i (1 if the sample belongs to class i, otherwise 0)
#pi : the predicted probability for class i
#n: the number of classes

# for example if the ouput layer has ouputs
softmax_output = np.array([[0.7,0.1,0.2],
                           [0.1,0.5,0.4],
                           [0.02,0.9,0.08]])

class_targets = [0,1,1]

print(softmax_output[[0,1,2], class_targets])

print(-np.log(softmax_output[range(len(softmax_output)),class_targets]))
neg_log = -np.log(softmax_output[range(len(softmax_output)),class_targets])
average_loss = np.mean(neg_log)

print(average_loss)