#include "../includes/layer.h"
#include <math.h>
#include <stdbool.h>

/* The sigmoid function and derivative. */
double sigmoid(double x)
{
  assert((1 + exp(-x)) != 0);
  return 1/(1 + exp(-x));
}

double sigmoidprime(double x)
{
  return x*(1 - x);
}

/* Creates a single layer. */
layer_t *layer_create()
{
    layer_t *layer = malloc(sizeof(layer_t));
    if (layer == NULL){
        perror("Layer Allocation");
        return NULL;
    }
    layer->num_inputs = 0;
    layer->num_outputs= 0;
    layer->outputs = NULL;
    layer->prev = NULL;
    layer->next = NULL;
    layer->weights = NULL;
    layer->biases = NULL;
    layer->deltas = NULL;
    return layer;
}

/* Initialises the given layer. */
bool layer_init(layer_t *layer, int num_outputs, layer_t *prev)
{
    assert(layer != NULL);
    if (prev == NULL){
        // layer is input layer
    }
    layer->prev = prev;
    layer->num_outputs = num_outputs;
    layer->outputs = malloc(num_outputs * sizeof(double));
    if (layer->outputs == NULL){
        perror("Outputs Allocation Error");
        return true;
    }
    for (int i = 0; i < num_outputs; i++)
        layer->outputs[i] = 0;
    if (prev != NULL){
        // Not the input layer
        layer->num_inputs = prev->num_outputs;
        // Allocate weights, biases and deltas arrays.
        // Weights, each output has input number of weights.
        layer->weights = malloc(sizeof(double) * prev->num_outputs * num_outputs);
        if (layer->weights == NULL){
            perror("Weights Allocation Error");
            return true;
        }
        for (int i = 0; i < prev->num_outputs; i++){
            layer->weights[i] = malloc(sizeof(double) * num_outputs);
            if (layer->weights[i] == NULL){
                perror("Weights Allocation Error");
                return true;
            }
            for (int j = 0; j < num_outputs; j++){
                layer->weights[i][j] = ANN_RANDOM();
            }
        }

        // Biases, each output has a bias.
        layer->biases = calloc(sizeof(double),num_outputs);
        if (layer->biases == NULL){
            perror("Biases Allocation Error");
            return true;
        }

        // Deltas, each output has a delta value.
        layer->deltas = calloc(sizeof(double), num_outputs);
        if (layer->deltas == NULL){
            perror("Deltas Allocation Error");
            return true;
        }
        for (int i = 0; i < num_outputs; i++){
            layer->biases[i] = 0;
            layer->deltas[i] = 0;
        }
    }
    return false;
}

/* Frees a given layer. */
void layer_free(layer_t *layer)
{
    assert(layer != NULL);
    if (layer->weights != NULL){
        // Free weights
        for (int i = 0; i < layer->num_inputs; i++){
            free(layer->weights[i]);
        }
        free(layer->weights);
    }
    if (layer->deltas != NULL){
        // Free deltas
        free(layer->deltas);
    }
    if (layer->biases != NULL){
        // Free biases
        free(layer->biases);
    }
    if (layer->outputs != NULL){
        // Free outputs
        free(layer->outputs);
    }
    free(layer);
}

/* Computes the outputs of the current layer. */
void layer_compute_outputs(layer_t const *layer)
{
   /* objective: compute layer->outputs */
    if (layer == NULL || layer->prev == NULL){
        return;
    }
    layer_compute_outputs(layer->prev);
    for (int j = 0; j < layer->num_outputs; ++j){
        double weight_sum = 0;

        // Sum of WijOi
        for (int i = 0; i < layer->num_inputs; ++i){
            weight_sum += layer->prev->outputs[i] * layer->weights[i][j];
        }

        layer->outputs[j] = sigmoid(layer->biases[j] + weight_sum);
    }
}

/* Computes the delta errors for this layer. */
// As of implementation, only output layer needs to be fed in.
void layer_compute_deltas(layer_t const *layer)
{
    if (layer == NULL || layer->next == NULL){
        return;
    }
    /* objective: compute layer->deltas */
    layer_compute_deltas(layer->next);

    for (int i = 0; i < layer->num_outputs; i++){
        double sum = 0;
        for (int j = 0; j < layer->next->num_outputs; j++){
            sum += layer->next->weights[i][j] * layer->next->deltas[j];
        }
        layer->deltas[i] = sum * sigmoidprime(layer->outputs[i]);
    }
}
/* Updates weights and biases according to the delta errors given learning rate. */
void layer_update(layer_t const *layer, double l_rate)
{
  /* objective: update layer->weights and layer->biases */
    if (layer == NULL || layer->prev == NULL) {
        return;
    }
    layer_update(layer->prev, l_rate);
    for (int i = 0; i < layer->num_inputs; i++){
        for (int j = 0; j < layer->num_outputs; j++){
            layer->weights[i][j] += l_rate *
                    layer->prev->outputs[i] * layer->deltas[j];
            layer->biases[j] += l_rate * layer->deltas[j];
        }
    }
}
