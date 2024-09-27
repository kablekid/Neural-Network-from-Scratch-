# we said  the output of the neuron is (inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2])  + bias
# this is just the dot prouct of inputs and weights  
# we can use numpy libarry to do .dot() product but we are goona implment a class that basically do the same thing but less efficent

class numpi:
    def __init__(self) -> None:
        self.output = 0
        self.outputarray = []


    def __isvalid__(self,value1:list,value2:list)->bool:
        
        if self.__check_input_type__(value1)[1] != self.__check_input_type__(value2)[0] :
            if  self.__check_input_type__(value1)[0] == 1 and self.__check_input_type__(value2)[0] == 1:
                return True
            raise ValueError("Shape Error")


    def __check_input_type__(self,matrix):
            # Check if it's a 1D list 
        if isinstance(matrix, list) and all(not isinstance(i, list) for i in matrix):
          
            return (1,len(matrix))  # 1D list so return the length
        
        # Check if it is a 2D matrix 
        elif isinstance(matrix, list) and all(isinstance(i, list) for i in matrix):
            rows = len(matrix)  # Number of rows
            cols = len(matrix[0]) if rows > 0 else 0  # Number of columns (assuming equal-length rows)
            return (rows, cols)  # 2D matrix so return (rows, cols)
        
        else:
            raise ValueError("Invalid matrix type. Must be a list or a list of lists.")
        



    def dot(self,value1:list[float],value2:list[float]) -> list[float]:

        self.__isvalid__(value1,value2) # check if operation possible  otherwise end it
        
        if  self.__check_input_type__(value1) == tuple((1,len(value1))) :
    
           if self.__check_input_type__(value2) == tuple((1,len(value2))) :
               #implement 1xn  1xn  dot product here return
                for i in range(min(len(value1),len(value2))):
                    self.output += value1[i] * value2[i]
                return self.output
               
           elif self.__check_input_type__(value2) == tuple((len(value2),len(value2[0]))) :
               #implement 1xn  nxn  dot product here return
               result = [0]*len(value2[0])
               for i in range(len(value2[0])):  
                for j in range(len(value1)):  
                    result[i] += value1[j] * value2[j][i]  
               return result
           
        elif self.__check_input_type__(value1) == tuple((len(value1),len(value1[0]))) :
            
            if self.__check_input_type__(value2) == tuple((1,len(value2))) :
                #implement nxn  nx1  dot product here return

                result = [0] * len(value2)
               
                for i in range(len(value2)):  
                    for j in range(len(value2[i])): 
                        result[i] += value2[i][j] * value1[j]  # Multiply row element by vector element

                return result
                
                pass
            elif self.__check_input_type__(value2) == tuple((len(value2),len(value2[0]))) :
                #implement nxn  nxn  dot product here return
                pass
    def t(self,matrix:list)->list:
        if not matrix or not isinstance(matrix[0], list):
            raise ValueError("Input must be a 2D list (matrix)")

        # Create a new matrix to hold the transposed values
        transposed = []
        
        # Iterate through columns
        for i in range(len(matrix[0])):  # Iterate over columns of the original matrix
            new_row = []
            for row in matrix:  # Iterate over each row
                new_row.append(row[i])  # Append the i-th element from each row
            transposed.append(new_row)  # Append the new row to the transposed matrix
        
        return transposed
                
        

if __name__ == "__main__":
    # Usage Example :
    inputs = [1.0,2.0,3.0,2.5]
    weights = [[0.2,0.8,-0.5,1.0],
            inputs,
            inputs,inputs]
    bias = 2.0


    numpyinstance = numpi()

    calculation = numpyinstance.dot(inputs,weights) 

    print(calculation)


    import numpy
    print(numpy.dot(inputs,weights))