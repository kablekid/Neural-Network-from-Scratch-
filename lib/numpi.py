class numpi(list):
    @staticmethod
    def check_shape(a, b):
        """Check if the shapes of two inputs are compatible for the dot product."""
        # If both are 1D lists (vectors)
        if isinstance(a[0], (int, float)) and isinstance(b[0], (int, float)):
            if len(a) != len(b):
                raise ValueError("Vectors must be of the same length for dot product.")
            return (len(a),)

        # If both are 2D lists (matrices)
        if isinstance(a[0], list) and isinstance(b[0], list):
            if len(a[0]) != len(b):
                raise ValueError("Matrix dimensions are not compatible for dot product.")
            return (len(a), len(b[0]))

        # If a is 1D (vector) and b is 2D (matrix)
        if isinstance(a[0], (int, float)) and isinstance(b[0], list):
            if len(a) != len(b):
                raise ValueError("Vector length must match the number of rows in the matrix.")
            return (len(b[0]),)

        # If a is 2D (matrix) and b is 1D (vector)
        if isinstance(a[0], list) and isinstance(b[0], (int, float)):
            if len(a[0]) != len(b):
                raise ValueError("Matrix columns must match vector length.")
            return (len(a),)

        raise ValueError("Invalid input types or dimensions.")

    @staticmethod
    def dot(a, b):
        """Perform the dot product of two vectors, matrices, or a combination of both."""
        # Check shapes for compatibility
        result_shape = numpi.check_shape(a, b)

        # Case 1: Dot product between two vectors (1D lists)
        if isinstance(a[0], (int, float)) and isinstance(b[0], (int, float)):
            return sum(a[i] * b[i] for i in range(len(a)))

        # Case 2: Dot product between two matrices (2D lists)
        if isinstance(a[0], list) and isinstance(b[0], list):
            result = [[0] * result_shape[1] for _ in range(result_shape[0])]
            for i in range(len(a)):
                for j in range(len(b[0])):
                    for k in range(len(b)):
                        result[i][j] += a[i][k] * b[k][j]
            return result

        # Case 3: Dot product between a vector and a matrix (1D list and 2D list)
        if isinstance(a[0], (int, float)) and isinstance(b[0], list):
            result = [0] * result_shape[0]
            for i in range(len(b[0])):  # Iterate over columns of the matrix
                for j in range(len(a)):  # Iterate over vector elements
                    result[i] += a[j] * b[j][i]
            return result

        # Case 4: Dot product between a matrix and a vector (2D list and 1D list)
        if isinstance(a[0], list) and isinstance(b[0], (int, float)):
            result = [0] * result_shape[0]
            for i in range(len(a)):  # Iterate over rows of the matrix
                for j in range(len(b)):  # Iterate over vector elements
                    result[i] += a[i][j] * b[j]
            return result

        raise ValueError("Dot product not supported for the given inputs.")
    
    @property
    def T(self):
        """Return the transpose of the matrix."""
        # Ensure it's a 2D list (matrix)
        if not isinstance(self.matrix[0], list):
            raise ValueError("Transpose is only applicable to 2D matrices.")
        
        # Transpose logic
        transposed = []
        for i in range(len(self.matrix[0])):  # Loop through columns
            new_row = [row[i] for row in self.matrix]  # Collect i-th elements from each row
            transposed.append(new_row)
        return transposed
    @staticmethod
    def addBias(output,bias):
        if len(output) != len(bias):
            raise ValueError("length must be the same")
        sum =[ output[i]+bias[i] for i in range(len(output))]
        return sum

