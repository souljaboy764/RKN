# Recurrent Kalman Networks

This is a fork of the original RKN repo at [LCAS](https://github.com/LCAS)/[RKN](https://github.com/LCAS/RKN) for testing its application to Motion Prediction and other areas. Please refer to the original repo for the ICML results

## Code

### n_link_sim
  Contains simulator for quad_link, see seperate readme
  
### rkn
#### data
Code to generate the needed data sets

### rkn
Implementation of the RKN as described in the paper
  - RKN: The full RKN consisting of encoder, RKNTransitionCell and decoder, implemented as subclass of keras.models.Model. Still abstract, the hidden structures of encoder and decoder need to be implemented for each experiment, see example experiments.
  - RKNTransitionCell the RKNTransitionCell as descirbed in the paper, implemented as sublcass of keras.layers.Layer in such a way that it can be used with keras.layers.RNN. 

### util
Utility functions

## Experiments

Currently Implemented:
  - Pendulum State Estimation (Implemented and verified that the ICML results are reproduced)
  - Pendulum Image Imputation (Implemented and verified that the ICML results are reproduced)
  - Quad Link State Estimation (Implemented)
  - Quad Link Image Imputation (Implemented)
  - Skeleton Motion Prediction (Implemented)


## Dependencies

Tested with:
  - python 3.6
  - tensorflow 2.6.0 (with GPU)
  - numpy 1.16
  - pillow 5.1.0 (only needed for data generation)

