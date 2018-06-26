#include "../includes/ann.h"

/* Creates and trains a simple ann for XOR. */
int main()
{
  printf("Big data machine learning.\n\n");
  printf("--------------------------\n");

  /* Intializes random number generator */
  srand(42);

  /* Here is some BIG DATA to train, XOR function. */
  
  int layer_outputs[] = {2, 2, 1};
  double batch_size = 1.0;
  int epoch_number = 25000; 
  const double inputs[4][2] = {{0, 0},
                               {0, 1},
                               {1, 0},
                               {1, 1}};
  const double targets[] = {0, 1, 1, 0};

  /* Create neural network. */
  printf("\n--------------------------\n");

  printf("%d inputs, %d hidden neurons and %d output.\n\n", layer_outputs[0], layer_outputs[1], layer_outputs[2]);
  printf(" * - * \\ \n");
  printf("         * - \n");
  printf(" * - * / \n\n");
  ann_t *xor_ann = ann_create(3, layer_outputs);
  if (!xor_ann) {
    printf("Couldn't create the neural network :(\n");
    return EXIT_FAILURE;
  }

  /* Initialise weights to random. */
  printf("Initialising network with random weights...\n");

  /* Print hidden layer weights, biases and outputs. */
  printf("The current state of the hidden layer:\n");
  for(int i=0; i < layer_outputs[0]; ++i) {
    for(int j=0; j < layer_outputs[1]; ++j)
      printf("  weights[%i][%i]: %f\n", i, j, xor_ann->input_layer->next->weights[i][j]);
  }
  for(int i=0; i < layer_outputs[1]; ++i)
    printf("  biases[%i]: %f\n", i, xor_ann->input_layer->next->biases[i]);
  for(int i=0; i < layer_outputs[1]; ++i)
    printf("  outputs[%i]: %f\n", i, xor_ann->input_layer->next->outputs[i]);

  /* Dummy run to see random network output. */
  printf("Current random outputs of the network:\n");
  for(int i = 0; i < 4; ++i) {
    ann_predict(xor_ann, inputs[i]);
    printf("  [%1.f, %1.f] -> %f\n", inputs[i][0], inputs[i][1], xor_ann->output_layer->outputs[0]);
  }

  /* Train the network. */
  printf("\nTraining the network...\n");
  for(int i = 0; i < epoch_number; ++i) {
    /* This is an epoch, running through the entire data. */
    for(int j = 0; j < 4; ++j) {
      /* Training at batch size 1, ie updating weights after every data point. */
      ann_train(xor_ann, inputs[j], targets + j, batch_size);
    }
  }

  /* Print hidden layer weights, biases and outputs. */
  printf("The current state of the hidden layer:\n");
  for(int i=0; i < layer_outputs[0]; ++i) {
    for(int j=0; j < layer_outputs[1]; ++j)
      printf("  weights[%i][%i]: %f\n", i, j, xor_ann->input_layer->next->weights[i][j]);
  }
  for(int i=0; i < layer_outputs[1]; ++i)
    printf("  biases[%i]: %f\n", i, xor_ann->input_layer->next->biases[i]);
  for(int i=0; i < layer_outputs[1]; ++i)
    printf("  outputs[%i]: %f\n", i, xor_ann->input_layer->next->outputs[i]);

  /* Let's see the results. */
  printf("\nAfter training magic happened the outputs are:\n");
  for(int i = 0; i < 4; ++i) {
    ann_predict(xor_ann, inputs[i]);
    printf("  [%1.f, %1.f] -> %f\n", inputs[i][0], inputs[i][1], xor_ann->output_layer->outputs[0]);
  }

  /* Time to clean up. */
  ann_free(xor_ann);

  return EXIT_SUCCESS;
}
