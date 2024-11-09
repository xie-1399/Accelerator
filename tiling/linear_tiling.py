import numpy as np


# Activation function definition
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Example constants (modify as needed)
Nn = 1024  # Number of output neurons
Ni = 1024  # Number of input neurons
Tnn = 64   # Block size for output neurons
Tii = 64   # Block size for input neurons
Tn = 16    # Small block size for output neurons
Ti = 16    # Small block size for input neurons

# Initialize data
synapse = np.full((Nn, Ni), 0.5)  # Weight matrix, filled with 0.5
neuron = np.ones(Ni)              # Input neurons, filled with 1
sum_array = np.zeros(Nn)           # Accumulator array


# Main computation function (blocked)
def neural_computation_blocked():
    global sum_array, neuron
    for nnn in range(0, Nn, Tnn):  # Tiling for large block of output neurons
        for iii in range(0, Ni, Tii):  # Tiling for large block of input neurons
            for nn in range(nnn, nnn + Tnn, Tn):  # Tiling for small block of output neurons
                sum_array[nn:nn + Tn] = 0  # Initialize the accumulator
                for ii in range(iii, iii + Tii, Ti):  # Tiling for small block of input neurons
                    for n in range(nn, nn + Tn):  # Iterate through the small block of output neurons
                        for i in range(ii, ii + Ti):  # Iterate through the small block of input neurons
                            sum_array[n] += synapse[n, i] * neuron[i]  # Calculate partial sum
                neuron[nn:nn + Tn] = sigmoid(sum_array[nn:nn + Tn])  # Apply activation function


# Direct matrix multiplication function
def neural_computation_direct():
    # Perform direct large matrix multiplication and apply activation function
    return sigmoid(np.dot(synapse, neuron))


# Initialize data
def initialize_data():
    global synapse, neuron
    neuron = np.ones(Ni)  # Initialize input neurons to 1
    synapse = np.full((Nn, Ni), 0.5)  # Initialize weight matrix to 0.5


# Main function
if __name__ == "__main__":
    initialize_data()  # Initialize inputs and weights

    # Blocked computation result
    neural_computation_blocked()  # Run blocked neural network computation
    result_blocked = neuron.copy()  # Save the result of blocked computation

    # Direct matrix multiplication result
    result_direct = neural_computation_direct()  # Run direct matrix multiplication

    # Output the comparison of the first 10 neurons
    print("Blocked computation results (first 10 neurons):")
    for i in range(10):
        print(f"neuron[{i}] = {result_blocked[i]}")

    print("\nDirect matrix multiplication results (first 10 neurons):")
    for i in range(10):
        print(f"neuron[{i}] = {result_direct[i]}")

    # Compare the difference between the two results
    difference = np.linalg.norm(result_blocked - result_direct)
    print(f"\nDifference between blocked computation and direct matrix multiplication (L2 norm): {difference}")
