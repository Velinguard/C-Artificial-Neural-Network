# C-Artificial-Neural-Network
Feed-Forward Artificial Neural Network entirely in C.

Utilising backpropagation to train a neural network. 

## Getting Started 
### Dependencies 
* C-99 compiler

### Installation

```
 git clone --repo
```

## Usage

### Using Included Training Set
In `testsuite/train.c` a sample implementation to solve the XOR problem can be found, this file can be edited to easily test the network.

#### Custom Input
To set input data edit the two dimensional array `inputs[][]` ensuring to update the size of the array in the form:

```
  inputs[num_inputs][num_input_nodes]
```

Change your input data accordingly.

#### Custom Output
To set corresponding outputs, edit the `targets[]` adding the associated output for each input, ensuring that the number of outputs is equal to the number of inputs.

#### Custom Layers
The layers are defined by the array `layer_outputs` where each integer states the number of nodes in each layer.

```
 layer_outputs[] = {num_input_nodes, num_hidden_nodes1, num_hidden_nodes2, ..., num_output_nodes};
```

#### Number of Epochs
To change the number of epochs simply update the variable `epoch_number` in `train.c`.

#### Batch Size
To change the batch size simply update the variable `batch_size` in `train.c`.


#### Train and Run
If you have edited the `train.c` file then you can just call:

``` 
  make
  .\network
 ```
The neural network will then train and show you the output.

### Using with your Code

#### Including Header Files
The following header files must be included:
```
 includes/ann.h
 includes/layer.h
```

#### Initialising Network
To initialise a neural network, create an array of integers of the form:
```
 {num_input_nodes, num_hidden_nodes1, num_hidden_nodes2, ..., num_output_nodes}
```
For example a neural network with 2 input nodes, one hidden layer with 2 nodes and 1 output node would be of the form:
```
 {2, 2, 1}
```
To create the neural network you must call 
```
 ann_create(int num_layers, int *layer_definition);
```
For example using the same network structure we would call
```
 ann_t *ann = ann_create(3, (int *) {2, 2, 1});
```
This will create an Artficial Neural Network with the correct structure and random weights. 

#### Outputing Result
To output the result of passing a value through the network, we use the following function

```
 ann_predict(ann_t *ann, int *inputs);
```

This will update the output nodes accordingly based upon the inputs array.

To get the outputs we must get the output layer.
```
 ann->output_layer;
```
An example of passing through an array of inputs and printing the result is as follows:
```
 ann_predict(ann, (int *) {1, 0});
 printf("  [1, 0] -> %f\n", ann->output_layer->outputs[0]);
```

#### Training the Neural Network

To train the network, we must use the following function
```
 ann_train(ann_t *ann, int *input_nodes, int *expected_output, int batch_size);
```
This function should be called for each input/output combination in the data set.

An example implementation of this using the XOR function is:
```
 for(int i = 0; i < epoch_number; ++i) {
  /* This is an epoch, running through the entire data. */
  for(int j = 0; j < 4; ++j) {
   /* Training at batch size 1, ie updating weights after every data point. */
   ann_train(ann, inputs[j], targets + j, batch_size);
  }
 }
```

#### Freeing Memory
After the network has been trained and utilised, you should free the memory associated using
```
 ann_free(ann_t *ann);
```


## Sources

For more information on Neural Networks and the maths behind Feed-Forward Neural Networks and Backpropagation visit my webpage at https://www.doc.ic.ac.uk/~sb3117.
