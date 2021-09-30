""" 
Python implementation of a simple Feedforward Neural Network

    Author: Gustavo Selbach Teixeira

"""
import random
import math


class Layer(object):

    def __init__(self, n_nodes, n_synapses=0):
        # initialize nodes
        self.values = [random.uniform(0, 1) for i in range(n_nodes)]
        self.bias = [random.uniform(0, 1) for i in range(n_nodes)]
        self.deltas = [random.uniform(0, 1) for i in range(n_nodes)]
        # initialize weights (synapses)
        self.weights = [[None] * n_nodes] * n_synapses
        for i in range(n_synapses):
            for j in range(n_nodes):
                self.weights[i][j] = random.uniform(0, 1)

class NeuralNetwork(object):
    learning_rate = 0.1

    def __init__(self, inputs, outputs, hidden=[]):
        self.input_layer = Layer(inputs, 0)
        #print(self.input_layer.nodes)
        last_size = inputs
        self.hidden_layers = []
        for hid in hidden:
            self.hidden_layers.append(Layer(hid, last_size))
            last_size = hid
        #print(self.input_layer.nodes)
        self.output_layer = Layer(outputs, last_size)
    
    def set_input(self, input_params):
        """ feed the network """
        assert len(input_params) == len(self.input_layer.values)
        
        for i in range(len(input_params)):
            self.input_layer.values[i] = input_params[i]
            
            
    def calc_delta_output(self, expected):
        """ calculate the deltas for the output layer """
        for i in range(len(self.output_layer.values)):
            error = (expected[i] - self.output_layer.values[i])
            self.output_layer.deltas[i] = (error 
                            * d_sigmoid(self.output_layer.values[i]))
    
    def train(self, inputs, outputs, n_iteractions):
        num_training_sets = len(outputs)
        training_sequence = list(range(num_training_sets))
        for n in range(n_iteractions):
            random.shuffle(training_sequence)
            for x in range(num_training_sets):
                i = training_sequence[x]
                self.set_input(inputs[i])
                
                activation_function(self.input_layer, self.hidden_layers[0])
                activation_function(self.hidden_layers[0], self.output_layer)

                print("Input: {} Expected: {} Output: {}".format(inputs[i],
                                                        outputs[i],
                                                        self.output_layer.values))

                self.calc_delta_output(outputs[i])
                calc_deltas(self.output_layer, self.hidden_layers[0])
                #calc_deltas(self.hidden_layers[0], self.input_layer)
            
                update_weights(self.output_layer, self.hidden_layers[0])
                update_weights(self.hidden_layers[0], self.input_layer)

    def predict(self, inputs):
        """ Make a prediction. To be used once the network is trained """
        self.set_input(inputs)
        activation_function(self.input_layer, self.hidden_layers[0])
        activation_function(self.hidden_layers[0], self.output_layer)
        return self.output_layer.values

def sigmoid(x):
    """ The logistical sigmoid function """
    return 1 / (1 + math.exp(-x))

def d_sigmoid(x):
    """ The derivative of sigmoid function """
    return x * (1 - x)

def activation_function(source, target):
    """ The Activation function """
    for j in range(len(target.values)):
        activation = target.bias[j]
        for k in range(len(source.values)):
            activation += (source.values[k] * target.weights[k][j])
        target.values[j] = sigmoid(activation)

def calc_deltas(source, target):
    n_nodes_source = len(source.values)
    for j in range(len(target.values)):
        error = 0.0
        for k in range(n_nodes_source):
            error += (source.deltas[k] * source.weights[j][k])
        target.deltas[j] = (error * d_sigmoid(target.values[j]))

def update_weights(source, target, learning_rate=0.1):
    target_length = len(target.values)
    for j in range(len(source.values)):
        source.bias[j] += (source.deltas[j] * learning_rate)
        for k in range(target_length):
            source.weights[k][j] += (target.values[k]
                                     * source.deltas[j] * learning_rate)


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
    #
    num_hidden_nodes = 4
    learning_rate = 0.1

    nn = NeuralNetwork(len(inputs[0]), len(outputs[0]), [num_hidden_nodes, ])

    nn.train(inputs, outputs, 10000)
    
    for i in range(len(inputs)):
        predicted = nn.predict(inputs[i])
        print("input: ", inputs[i],
              "predicted:", predicted,
              "output:", outputs[i])




