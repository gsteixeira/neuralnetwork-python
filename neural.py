""" 
A simple feed forward neural network in Python.
This version allows multiple hidden layers.


    Author: Gustavo Selbach Teixeira

"""
import random
import math


class Layer(object):
    """
    The layer object class.

    Attributes:
    -----------
    values : list
        This is where we write and read data from the NN.
    bias : list
        Used to compute the predicitons.
    deltas : list
        Used to correct the errors though the network.
    weights : list of lists - matrix
        Set the weights of the connections
    """
    def __init__(self, n_nodes:int, n_synapses:int=0):
        """ setup nn layers """
        self.values = [random.uniform(0, 1) for i in range(n_nodes)]
        self.bias = [random.uniform(0, 1) for i in range(n_nodes)]
        self.deltas = [random.uniform(0, 1) for i in range(n_nodes)]
        # initialize weights (synapses)
        self.weights = [[None] * n_nodes] * n_synapses
        for i in range(n_synapses):
            for j in range(n_nodes):
                self.weights[i][j] = random.uniform(0, 1)

class NeuralNetwork(object):
    """
    The Neural Network object. Holds the layers.

    Attributes:
    -----------
        input_layer : Layer
            Where data is inserted to the network.
        hidden_layers : list of Layers
            The list of Hidden Layers.
        output_layer : None
            Where data is read.
        learning_rate : 0.1
            The rate of the learning process.
    """
    learning_rate = 0.1

    def __init__(self, inputs:list, outputs:list, hidden:list):
        """ setup nn layers """
        self.input_layer = Layer(inputs, 0)
        self.hidden_layers_number = len(hidden)
        last_size = inputs
        self.hidden_layers = []
        for hid in hidden:
            self.hidden_layers.append(Layer(hid, last_size))
            last_size = hid
        self.output_layer = Layer(outputs, last_size)
    
    def set_input(self, input_params:list):
        """ Feed the network with data. """
        assert len(input_params) == len(self.input_layer.values)
        for i in range(len(input_params)):
            self.input_layer.values[i] = input_params[i]

    def activation_function(self, source, target):
        """ The Activation function """
        for j in range(len(target.values)):
            activation = target.bias[j]
            for k in range(len(source.values)):
                activation += (source.values[k] * target.weights[k][j])
            target.values[j] = sigmoid(activation)

    def calc_delta_output(self, expected:list):
        """ Compute the deltas for the output layer """
        for i in range(len(self.output_layer.values)):
            error = (expected[i] - self.output_layer.values[i])
            self.output_layer.deltas[i] = (error 
                            * d_sigmoid(self.output_layer.values[i]))

    def calc_deltas(self, source, target):
        """ Compute the deltas between layers """
        n_nodes_source = len(source.values)
        for j in range(len(target.values)):
            error = 0.0
            for k in range(n_nodes_source):
                error += (source.deltas[k] * source.weights[j][k])
            target.deltas[j] = (error * d_sigmoid(target.values[j]))

    def update_weights(self, source, target):
        """ Update the weights """
        target_length = len(target.values)
        for j in range(len(source.values)):
            source.bias[j] += (source.deltas[j] * self.learning_rate)
            for k in range(target_length):
                source.weights[k][j] += (target.values[k]
                                        * source.deltas[j] * self.learning_rate)

    def forward_pass(self):
        """ NN Activation step """
        k = 0
        self.activation_function(self.input_layer, self.hidden_layers[k])
        # Run through the hidden layers. If theres more than 1.
        last_hidden_layer = nn.hidden_layers_number - 1
        while k < last_hidden_layer:
            self.activation_function(self.hidden_layers[k],
                                     self.hidden_layers[k+1])
            k += 1
        self.activation_function(self.hidden_layers[k], self.output_layer)

    def back_propagation(self, outputs:list):
        """ The back propagation process.
        Computes the deltas and update the weights and bias.
        If there' multiple hidden_layers, loops though then back and forth
        """
        self.calc_delta_output(outputs)
        k = nn.hidden_layers_number - 1
        self.calc_deltas(self.output_layer, self.hidden_layers[k])
        self.update_weights(self.output_layer, self.hidden_layers[k])
        while k > 0:
            self.calc_deltas(self.hidden_layers[k], self.hidden_layers[k-1])
            self.update_weights(self.hidden_layers[k],
                                self.hidden_layers[k-1])
            k -= 1
        self.update_weights(self.hidden_layers[k], self.input_layer)

    def train(self, inputs:list, outputs:list, n_iteractions:int):
        """ Training main loop """
        num_training_sets = len(outputs)
        training_sequence = list(range(num_training_sets))
        for n in range(n_iteractions):
            random.shuffle(training_sequence)
            for x in range(num_training_sets):
                i = training_sequence[x]
                self.set_input(inputs[i])
                # forward activation
                self.forward_pass()
                print("Input: {} Expected: {} Output: {}".format(inputs[i],
                                                    outputs[i],
                                                    self.output_layer.values))
                self.back_propagation(outputs[i])

    def predict(self, inputs:list):
        """ Make a prediction. To be used after the network is trained """
        self.set_input(inputs)
        self.forward_pass()
        return self.output_layer.values

def sigmoid(x):
    """ The logistical sigmoid function """
    return 1 / (1 + math.exp(-x))

def d_sigmoid(x):
    """ The derivative of sigmoid function """
    return x * (1 - x)

if __name__ == "__main__":
    # Training parameters
    inputs = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]]
    outputs = [[0.0],
               [1.0],
               [1.0],
               [0.0]]
    # Set up the network
    hidden_nodes = [4,]
    nn = NeuralNetwork(len(inputs[0]), len(outputs[0]), hidden_nodes)
    nn.learning_rate = 0.1
    nn.train(inputs, outputs, 10000)
    # Now the network is fit, lets try some predictions
    for i in range(len(inputs)):
        predicted = nn.predict(inputs[i])
        print("input: ", inputs[i],
              "output:", outputs[i],
              "predicted: {:.2f}".format(predicted[0]))

