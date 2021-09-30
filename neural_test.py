
from neural_oo import NeuralNetwork
from neural_oo import activation_function, calc_deltas, update_weights
import pytest

num_inputs = 2
num_hidden_nodes = 2
num_outputs = 1
learning_rate = 0.1


nn = NeuralNetwork(num_inputs, num_outputs, [num_hidden_nodes, ])

#@pytest.mark.skip
def test_activation_function():
    #source_layer =  [1.0, 0.0]
    source_layer =  [0.0, 1.0]
    target_layer =  [0.1, 0.2]
    target_layer_bias =  [0.1, 0.2]
    weights =  [[0.1, 0.2], [0.3, 0.4]]
    nn.input_layer.values = source_layer
    nn.hidden_layers[0].bias = target_layer_bias
    nn.hidden_layers[0].values = target_layer
    nn.hidden_layers[0].weights = weights
    
    activation_function(nn.input_layer, nn.hidden_layers[0])
    #TARGET_LAYER_FINAL =  [0.549833997312478, 0.598687660112452] # 1,0
    TARGET_LAYER_FINAL = [0.598687660112452, 0.6456563062257954] # 0,1
    for i in range(len(nn.hidden_layers[0].values)):
        assert nn.hidden_layers[0].values[i] == TARGET_LAYER_FINAL[i]


#@pytest.mark.skip
def test_calc_deltas():
    
    source_layer =  [0.1]
    delta_source =  [0.2]
    target_layer =  [0.3, 0.4]
    source_weights =  [[0.5, None], [0.6, None]]

    #source = nn.output_layer
    nn.output_layer.values = source_layer.copy()
    nn.output_layer.deltas = delta_source.copy()
    nn.output_layer.weights = source_weights.copy()
    
    nn.hidden_layers[0].values = target_layer.copy()

    deltas = calc_deltas(nn.output_layer, nn.hidden_layers[0])

    DELTA_FINAL = [0.021, 0.0288]
    for i in range(len(DELTA_FINAL)):
        print(nn.hidden_layers[0].deltas[i], DELTA_FINAL[i])
        assert nn.hidden_layers[0].deltas[i] == DELTA_FINAL[i]
    

#@pytest.mark.skip
def test_calc_delta_output_layer():
    expected = [1.0]
    output_layer = [0.1]
    
    source = nn.output_layer
    for i in range(len(nn.output_layer.values)):
        nn.output_layer.values[i] = output_layer[i]
    
    #calc_delta_output(expected, nn.output_layer)
    nn.calc_delta_output(expected)
    DELTA_FINAL =  [0.08100000000000002]
    
    for i in range(len(DELTA_FINAL)):
        assert nn.output_layer.deltas[i] == DELTA_FINAL[i]

#@pytest.mark.skip
def test_update_weights():
    source_bias =  [0.1, 0.2]
    target_layer =  [1.0, 0.0]
    weights =  [[0.1, 0.2], [0.3, 0.4]]
    deltas =  [0.1, 0.2]

    nn.hidden_layers[0].deltas = deltas.copy()
    nn.hidden_layers[0].bias = source_bias.copy()
    nn.hidden_layers[0].weights = weights.copy()
    nn.input_layer.values = target_layer.copy()

    learning_rate = 0.1
    update_weights(nn.hidden_layers[0], nn.input_layer, learning_rate)
    ###
    SOURCE_BIAS_FINAL = [0.11000000000000001, 0.22000000000000003]
    WEIGHTS_FINAL =  [[0.11000000000000001, 0.22000000000000003], [0.3, 0.4]]

    for i in range(len(SOURCE_BIAS_FINAL)):
        assert nn.hidden_layers[0].bias[i] == SOURCE_BIAS_FINAL[i]

    for i in range(len(WEIGHTS_FINAL)):
        for j in range(len(WEIGHTS_FINAL[i])):
            assert nn.hidden_layers[0].weights[i][j] == WEIGHTS_FINAL[i][j]

if __name__ == "__main__":
    test_activation_function()
    test_calc_deltas()
    test_calc_delta_output_layer()
    test_update_weights()
