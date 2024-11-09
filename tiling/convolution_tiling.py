import numpy as np

'''
here is a simple 2-D conv example
'''


def convolution_with_relu(input_matrix, kernel, block_size):
    input_h, input_w = input_matrix.shape
    kernel_h, kernel_w = kernel.shape
    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1
    output_matrix = np.zeros((output_h, output_w))

    # Perform convolution calculation in blocks
    for i in range(0, output_h, block_size):
        for j in range(0, output_w, block_size):
            # Calculate the boundaries of the current block
            block_h_end = min(i + block_size, output_h)
            block_w_end = min(j + block_size, output_w)

            # Perform convolution within the current block
            for m in range(i, block_h_end):
                for n in range(j, block_w_end):
                    region = input_matrix[m:m + kernel_h, n:n + kernel_w]
                    output_matrix[m, n] = np.sum(region * kernel)

    # Apply ReLU function
    output_matrix = np.maximum(output_matrix, 0)

    return output_matrix


if __name__ == '__main__':
    # Example input
    input_matrix = np.array([[1, 2, 3, 0],
                             [4, 5, 6, 1],
                             [7, 8, 9, 0],
                             [1, 3, 5, 2]])

    kernel = np.array([[1, 0],
                       [0, -1]])

    # Call the convolution function with a block size of 2
    output = convolution_with_relu(input_matrix, kernel, block_size=2)
    print("Output after convolution (with ReLU):")
    print(output)
