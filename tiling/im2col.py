import numpy as np

'''
through the img2col you can convert the convolution operation to the GEMM operation
'''


def im2col(input_matrix, kernel_h, kernel_w):
    input_h, input_w = input_matrix.shape
    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1

    # Initialize an empty matrix to store the unrolled columns
    col_matrix = np.zeros((kernel_h * kernel_w, output_h * output_w))

    col_index = 0
    for i in range(output_h):
        for j in range(output_w):
            region = input_matrix[i:i + kernel_h, j:j + kernel_w].reshape(-1)
            col_matrix[:, col_index] = region
            col_index += 1

    return col_matrix


def convolution_via_gemm(input_matrix, kernel):
    kernel_h, kernel_w = kernel.shape
    input_h, input_w = input_matrix.shape

    # Convert input matrix to column matrix using im2col
    col_matrix = im2col(input_matrix, kernel_h, kernel_w)

    # Reshape kernel to a row vector
    kernel_vector = kernel.reshape(1, -1)

    # Perform matrix multiplication (GEMM)
    output_vector = np.dot(kernel_vector, col_matrix)

    # Reshape the output to the correct shape
    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1
    output_matrix = output_vector.reshape(output_h, output_w)

    return output_matrix


# Example input
if __name__ == '__main__':
    input_matrix = np.array([[1, 2, 3, 0],
                             [4, 5, 6, 1],
                             [7, 8, 9, 0],
                             [1, 3, 5, 2]])

    kernel = np.array([[1, 0],
                       [0, -1]])

    # Call the convolution function via GEMM
    output = convolution_via_gemm(input_matrix, kernel)
    print("Output after convolution via GEMM:")
    print(output)
