#include "ann.h"

/* Creates and returns a new ann. */
ann_t *ann_create(int num_layers, int *layer_outputs)
{
    assert(layer_outputs != NULL);
    assert(num_layers > 0);
    ann_t *nn = malloc(sizeof(ann_t));
    if (nn == NULL){
        perror("Neural Network Allocation Error");
        return NULL;
    }
    if ((nn->input_layer = layer_create()) == NULL){
        return NULL;
    }
    if (layer_init(nn->input_layer, layer_outputs[0], NULL)){
        return NULL;
    }
    layer_t *last_nn = nn->input_layer;
    for (int i = 1; i < num_layers; i++){
        layer_t *nn_temp;
        if ((nn_temp = layer_create()) == NULL){
            return NULL;
        }
        if (layer_init(nn_temp, layer_outputs[i], last_nn)){
            return NULL;
        }
        last_nn->next = nn_temp;
        last_nn = nn_temp;
    }

    nn->output_layer = last_nn;
    return nn;
}

/* Frees the space allocated to ann. */
void ann_free(ann_t *ann)
{
    assert(ann != NULL);
    assert(ann->output_layer != NULL);
    layer_t *layer = ann->output_layer;
    while(layer != NULL){
        layer_t *prev_layer = layer->prev;
        layer_free(layer);
        layer = prev_layer;
    }
    free(ann);
}

/* Forward run of given ann with inputs. */
void ann_predict(ann_t const *ann, double const *inputs)
{
    assert(ann != NULL && inputs != NULL && ann->input_layer != NULL);
    assert(ann->output_layer != NULL);
    assert(sizeof(inputs) >= ann->input_layer->num_outputs);
    for (int i = 0; i < ann->input_layer->num_outputs; i++){
        ann->input_layer->outputs[i] = inputs[i];
    }
    layer_compute_outputs(ann->output_layer);
}

/* Trains the ann with single backprop update. */
void ann_train(ann_t const *ann, double const *inputs, double const *targets, double l_rate)
{
    /* Sanity checks. */
    assert(ann != NULL);
    assert(inputs != NULL);
    assert(targets != NULL);
    assert(l_rate > 0);

    /* Run forward pass. */
    ann_predict(ann, inputs);

    for (int i = 0; i < ann->output_layer->num_outputs; i++){
        ann->output_layer->deltas[i] =
            sigmoidprime(ann->output_layer->outputs[i]) *
                    (targets[i] - ann->output_layer->outputs[i]);
    }
    layer_compute_deltas(ann->output_layer->prev);
    layer_update(ann->output_layer, l_rate);
}
