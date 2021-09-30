
from neural import (activation_function, calc_deltas,
                    calc_delta_output, update_weights, train)
import pytest

#@pytest.mark.skip
def test_activation_function():
    source_layer =  [0.0, 1.0]
    target_layer =  [0.1, 0.2]
    target_bias =  [0.1, 0.2]
    weights =  [[0.1, 0.2], [0.3, 0.4]]
    
    print(weights)
    activation_function(source_layer=source_layer,
                        target_layer=target_layer,
                        target_bias=target_bias,
                        target_weights=weights)
    TARGET_LAYER_FINAL = [0.549833997312478, 0.598687660112452] # 1, 0
    TARGET_LAYER_FINAL = [0.598687660112452, 0.6456563062257954] # 0, 1
    
    print(target_layer)
    for i in range(len(TARGET_LAYER_FINAL)):
        print(target_layer[i])
        assert target_layer[i] == TARGET_LAYER_FINAL[i]

#@pytest.mark.skip
def test_calc_deltas():
    source_layer =  [0.1]
    delta_source =  [0.2]
    target_layer =  [0.3, 0.4]
    source_weights =  [[0.5, None], [0.6, None]]
    deltas = calc_deltas(source_layer=source_layer,
                         source_delta=delta_source,
                         target_layer=target_layer,
                         source_weights=source_weights)
    DELTA_FINAL = [0.021, 0.0288]
    for i in range(len(DELTA_FINAL)):
        assert deltas[i] == DELTA_FINAL[i]
    

#@pytest.mark.skip
def test_calc_delta_output_layer():
    expected = [1.0]
    output_layer = [0.1]
    
    deltas = calc_delta_output(expected, output_layer)
    DELTA_FINAL =  [0.08100000000000002]
    for i in range(len(DELTA_FINAL)):
        assert deltas[i] == DELTA_FINAL[i]

#@pytest.mark.skip
def test_update_weights():
    source_bias =  [0.1, 0.2]
    target_layer =  [1.0, 0.0]
    weights =  [[0.1, 0.2], [0.3, 0.4]]
    delta =  [0.1, 0.2]
    learning_rate = 0.1
    
    update_weights(source_bias=source_bias,
                   source_weights=weights,
                   source_delta=delta,
                   target_layer=target_layer,
                   learning_rate=learning_rate)
    ##
    SOURCE_BIAS_FINAL = [0.11000000000000001, 0.22000000000000003]
    WEIGHTS_FINAL =  [[0.11000000000000001, 0.22000000000000003], [0.3, 0.4]]
    for i in range(len(SOURCE_BIAS_FINAL)):
        assert source_bias[i] == SOURCE_BIAS_FINAL[i]
    for i in range(len(WEIGHTS_FINAL)):
        assert weights[i] == WEIGHTS_FINAL[i]


def test_train():
    inputs = [[0.0, 0.0],
              [1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0]]
    outputs = [[0.0],
               [1.0],
               [1.0],
               [0.0]]
    iteracions = 1
    train(inputs, outputs, iteracions, 1)
    
    
if __name__ == "__main__":
    test_activation_function()
    test_train()
    test_calc_deltas()


