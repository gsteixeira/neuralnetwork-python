# A Neural Network in Python

A feed forward neural network in Python


## usage:

If you just wanna try it:
```shell
    python neural.py
```

## Create a neural network:

Create a network telling the size (nodes) of earch layer.
```python
    nn = NewNeuralNetwork(size_input,
                           size_output,
                           [hidden_sizes,])
    # train the network
    nn.train(inputs, outputs, iteractions)
    # Make predictions
    predicted = nn.predict(inputs[i])
```

## Two implementations:

There are two implementations on this repository:
- neural.py - Object oriented, more flexible and has more features.
- neural_procedural.py - Simple, procedural implentation that uses arrays.




