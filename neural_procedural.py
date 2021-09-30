""" 
Python implementation of a simplistic Feedforward Neural Network

    Author: Gustavo Selbach Teixeira

 Procedural simplistic implementation. 
 Uses only arrays instead of objects
"""
import math
import random

def sigmoid(x):
    """ The logistical sigmoid function """
    return 1 / (1 + math.exp(-x))

def d_sigmoid(x):
    """ The derivative of sigmoid function """
    return x * (1 - x)

def relu(x):
    """ The rectified linear unit """
    return max(0, x)

def new_layer(size, parent_size):
    """ Initialize the vectors for a new layer """
    layer = [None] * size
    layer_bias = [None] * size
    layer_weights = [[None] * size] * parent_size
    for i in range(size):
        layer[i] = random.uniform(0, 1)
        layer_bias[i] = random.uniform(0, 1)
        for j in range(parent_size):
            layer_weights[j][i] = random.uniform(0, 1)
    return layer, layer_bias, layer_weights

def activation_function(source_layer=[], target_layer=[],
                        target_bias=[], target_weights=[]):
    """ The Activation function """
    for j in range(len(target_layer)):
        activation = target_bias[j]
        for i in range(len(source_layer)):
            activation += (source_layer[i] * target_weights[i][j])
        target_layer[j] = sigmoid(activation)

def calc_deltas(source_layer=[], source_delta=[],
                target_layer=[], source_weights=[]):
    """ Calculate the Deltas """
    num_nodes_target = len(target_layer)
    num_nodes_source = len(source_layer)
    delta = [None] * num_nodes_target
    for j in range(num_nodes_target):
        error = 0.0
        for i in range(num_nodes_source):
            error += (source_delta[i] * source_weights[j][i])
        delta[j] = (error * d_sigmoid(target_layer[j]))
    return delta

def calc_delta_output(expected, output_layer):
    """ Calculate the delta for the output layer """
    delta_output = [None] * len(output_layer)
    for j in range(len(output_layer)):
        error_output = (expected[j] - output_layer[j])
        delta_output[j] = (error_output * d_sigmoid(output_layer[j]))
    return delta_output

def update_weights(source_bias=[], source_weights=[], 
                   source_delta=[], target_layer=[], learning_rate=0.1):
    """ Update the weights of synapses """
    dest_number = len(target_layer)
    for j in range(len(source_bias)):
        source_bias[j] += (source_delta[j] * learning_rate)
        for i in range(dest_number):
            source_weights[i][j] += (target_layer[i] 
                                     * source_delta[j] * learning_rate)

def train(inputs, outputs, n_epochs):
    """ Neural network training loop """
    num_inputs = len(inputs[0])
    num_outputs = len(outputs[0])
    num_training_sets = len(outputs)
    
    num_hidden_nodes = 4
    learning_rate = 0.1

    hit_counter = 0
    miss_counter = 0

    # Initialize the vectors for the hidden layer
    hidden_layer, hidden_bias, hidden_weights = new_layer(
        num_hidden_nodes, num_inputs)

    # Initialize the vectors for the output layer
    output_layer, output_bias, output_weights = new_layer(
        num_outputs, num_hidden_nodes)

    training_sequence = list(range(num_training_sets))
    for n in range(n_epochs):
        # shuffles the order of the training set to increase entropy
        random.shuffle(training_sequence)
        for x in range(num_training_sets):
            i = training_sequence[x]
            # Forward pass
            activation_function(source_layer=inputs[i],
                                target_layer=hidden_layer,
                                target_bias=hidden_bias,
                                target_weights=hidden_weights)

            activation_function(source_layer=hidden_layer,
                                target_layer=output_layer,
                                target_bias=output_bias,
                                target_weights=output_weights)
            print("Input: {} Expected: {} Output: {}".format(inputs[i],
                                                             outputs[i],
                                                             output_layer))
            # Back propagation
            # calculate delta for output_layer
            delta_output = calc_delta_output(outputs[i], output_layer)
            # calculate delta from output to hidden_layer
            delta_hidden = calc_deltas(source_layer=output_layer,
                                        source_delta=delta_output,
                                        target_layer=hidden_layer,
                                        source_weights=output_weights)
            # Update weights
            # from Output to hidden
            update_weights(source_bias=output_bias,
                            source_weights=output_weights,
                            source_delta=delta_output,
                            target_layer=hidden_layer,
                            learning_rate=learning_rate)

            # from hidden to input layer
            update_weights(source_bias=hidden_bias,
                            source_weights=hidden_weights,
                            source_delta=delta_hidden,
                            target_layer=inputs[i],
                            learning_rate=learning_rate)


if __name__ == "__main__":
    inputs = [[0.0, 0.0],
              [1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0]]
    outputs = [[0.0],
               [1.0],
               [1.0],
               [0.0]]
    iteracions = 10000
    train(inputs, outputs, iteracions)
