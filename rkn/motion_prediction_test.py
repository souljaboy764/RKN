import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from sklearn.model_selection import train_test_split
import sys

import argparse

from rkn.RKN import RKN
from rkn.RKNTransitionCell import RKNTransitionCell
from util.MVNXDataDriver import *
from networks.mopred import *


parser = argparse.ArgumentParser(description='Train Autoencoder network')
parser.add_argument('--src-dir', type=str, metavar='SRC', required=True,
                    help='Directory to the npy files where the skeletons are stored.')
parser.add_argument('--model-dir', type=str, metavar='DIR', required=True,
                    help='Directory where the model files are saved.')
parser.add_argument('--seq-len', type=int, default=100, metavar='LEN',
                    help='Sequences length (default: 100)')
args = parser.parse_args()

# Read Data
obs, targets = generate_motion_prediction_set(args.src_dir, args.seq_len)
train_obs, test_obs, train_targets, test_targets = train_test_split(obs, targets, test_size=0.2, random_state=42)

# Build Model
saved_model = k.models.load_model(args.model_dir, custom_objects = {"RKN":RKN, "RKNTransitionCell": RKNTransitionCell, "rmse":RKN.rmse})

rkn = MVNXMotionPredictionRKN(observation_shape=train_obs.shape[-1], latent_observation_dim=saved_model._enc_hidden_layers[-1].layer.units//2, hidden_layer_dim=saved_model._enc_hidden_layers[0].layer.units,
								 output_dim=train_targets.shape[-1], num_basis=15, bandwidth=3, never_invalid=True, batch_size=test_obs.shape[0])

rkn.set_weights(saved_model.get_weights())

test_preds = rkn.predict(test_obs)

np.save('rkn_mopred.npy', {"test_obs":test_obs, "test_preds":test_preds, "test_targets":test_targets})

# Iteratively predict with observing 10%, 20% and 50% of the trajectory
test_obs_10 = test_obs.copy()
test_obs_20 = test_obs.copy()
test_obs_50 = test_obs.copy()

test_preds_10 = test_preds.copy()
test_preds_20 = test_preds.copy()
test_preds_50 = test_preds.copy()

rkn._layer_rkn.reset_states()
test_preds_10[:, :int(0.1*args.seq_len), :] = rkn.predict(test_obs_10[:, :int(0.1*args.seq_len), :])

for idx in range(int(0.1*args.seq_len), args.seq_len - 1):
	test_obs_10[:, idx, :] = test_preds_10[:, idx - 1, :train_targets.shape[-1]]
	test_preds_10[:, idx:idx+1, :] = rkn.predict(test_obs_10[:, idx:idx+1, :])

np.save('rkn_mopred_10.npy', {"test_obs":test_obs, "test_preds":test_preds_10, "test_targets":test_targets})


rkn._layer_rkn.reset_states()
test_preds_20[:, :int(0.2*args.seq_len), :] = rkn.predict(test_obs_20[:, :int(0.2*args.seq_len), :])

for idx in range(int(0.2*args.seq_len), args.seq_len - 1):
	test_obs_20[:, idx, :] = test_preds_20[:, idx - 1, :train_targets.shape[-1]]
	test_preds_20[:, idx:idx+1, :] = rkn.predict(test_obs_20[:, idx:idx+1, :])

np.save('rkn_mopred_20.npy', {"test_obs":test_obs, "test_preds":test_preds_20, "test_targets":test_targets})

rkn._layer_rkn.reset_states()
test_preds_50[:, :int(0.5*args.seq_len), :] = rkn.predict(test_obs_50[:, :int(0.5*args.seq_len), :])

for idx in range(int(0.5*args.seq_len), args.seq_len - 1):
	test_obs_50[:, idx, :] = test_preds_50[:, idx - 1, :train_targets.shape[-1]]
	test_preds_50[:, idx:idx+1, :] = rkn.predict(test_obs_50[:, idx:idx+1, :])

np.save('rkn_mopred_50.npy', {"test_obs":test_obs, "test_preds":test_preds_50, "test_targets":test_targets})