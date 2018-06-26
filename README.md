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

### Custom Training Set
In `testsuite/train.c` a sample implementation to solve the XOR problem can be found.

#### Custom Input
To set input data edit the two dimensional array `inputs[][]` ensuring to update the size of the array in the form:

```
  inputs[num_inputs][num_input_nodes]
```

Change your input data accordingly.

#### Custom Output
To set corresponding outputs, edit the `targets[]` adding the associated output for each input, ensuring that the number of outputs is equal to the number of inputs.

#### Number of Epochs
To change the number of epochs simply update the variable `epoch_number` in `train.c`.

#### Batch Size
To change the batch size simply update the variable `batch_size` in `train.c`.

### Train
If you have edited just the `train.c` file then you can just call:

``` 
  make
```

However if you have used a different file, you will need to edit the `makefile` accordingly. Changing the text containing train.c with your new filename.

If you use make you can call: 

 ```
  .\network
 ```
The neural network will then train and show you the output.

## Sources

For more information on Neural Networks and the maths behind Feed-Forward Neural Networks and Backpropagation visit my webpage at https://www.doc.ic.ac.uk/~sb3117.
