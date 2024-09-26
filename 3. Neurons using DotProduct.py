# we said  the output of the neuron is (inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2])  + bias
# this is just the dot prouct of inputs and weights  
# we can use numpy libarry to do .dot() product but we are goona implment a function that basically do the same thing

class numpy:
    def __init__(self) -> None:
        output = 0
        outputarray = []

    def __check_input_type__(self,matrix):
        if isinstance(matrix, list):
            if all(isinstance(i, list) for i in matrix):
                if len(matrix) == 1:  # 1D Matrix (list of lists with a single row)
                    return 1
                else:  # 2D Matrix
                    return 2
            else:  # 1D List
                return 1
        else:
            raise ValueError("Invalid input type: must be a list or list of lists")
        

    def dot(self,value1:list[float],value2:list[float]) -> list[float]:
        if  self.__check_input_type__(value1) == 1:
            if self.__check_input_type__(value2) == 1:
                for i in range(len(value1)):
                    output += value1[i] * value2[i]
                return output
            
            elif self.__check_input_type__(value2) == 2:
                for values in value2:
                    for i,value in enumerate(values):
                        output += value*value1[i]
                    
                    self.outputarray.append(output)
                    output = 0
                return self.outputarray







inputs = [1,2,3]
weights = [0.9,0.5,-0.3]
bias = 1


numpy = numpy()

x = numpy.dot(input,weights)

print(x)