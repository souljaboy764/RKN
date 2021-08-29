import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from sklearn.model_selection import train_test_split
import sys

import argparse

from rkn.RKN import RKN
from rkn.RKNTransitionCell import RKNTransitionCell
from util.LayerNormalization import LayerNormalization
from util.MVNXDataDriver import *


parser = argparse.ArgumentParser(description='Train Autoencoder network')
parser.add_argument('--src-dir', type=str, metavar='SRC', required=True,
                    help='Directory to the npy files where the skeletons are stored.')
parser.add_argument('--model-dir', type=str, metavar='DIR', required=True,
                    help='Directory where the model files are saved.')
parser.add_argument('--hidden-dim', type=int, default=50, metavar='H',
                    help='Dimensions of the hidden layers (default: 50).')
parser.add_argument('--latent-dim', type=int, default=5, metavar='L',
                    help='Dimensions of the latent space (default: 5).')

parser.add_argument('--seq-len', type=int, default=540, metavar='LEN',
                    help='Sequences length (default: 540)')
args = parser.parse_args()

def generate_motion_prediction_set(root_dir, seq_length):
	DATA_PARAMS = {}
	DATA_PARAMS.update({"data_source": "MVNX", "nb_frames":seq_length, 'as_3D': True, 'data_types': ['position'],"unit_bounds": False, "path":root_dir})
	data_driver = MVNXDataDriver(DATA_PARAMS)
	data_driver.parse(frameMod=True)
	shape = data_driver.data.shape
	data = data_driver.data.reshape(-1, shape[-2], shape[-1])
	return data[:, :-1, :], data[:, 1:, :]


# Implement Encoder and Decoder hidden layers
class MVNXMotionPredictionRKN(RKN):

	def build_encoder_hidden(self):
		return [
			# 1: Dense Layer (Input to hidden)
			k.layers.Dense(args.hidden_dim, activation=k.activations.relu),
			# 2: Dense Layer (hidden to latent)
			k.layers.Dense(2*self._lod, activation=k.activations.relu)
		]

	def build_decoder_hidden(self):
		return [
			# 1: Dense Layer (latent to hidden)
			k.layers.Dense(args.hidden_dim, activation=k.activations.relu),
			# 2: Dense Layer (hidden to Input)
			k.layers.Dense(self._output_dim, activation=k.activations.relu)
		]
	
	def build_var_decoder_hidden(self):
		return [
			# 1: Dense Layer (latent to hidden)
			k.layers.Dense(args.hidden_dim, activation=k.activations.relu),
			# 2: Dense Layer (hidden to Input)
			k.layers.Dense(self._output_dim, activation=k.activations.relu)
		]

# Read Data
obs, targets = generate_motion_prediction_set(args.src_dir, args.seq_len)
train_obs, test_obs, train_targets, test_targets = train_test_split(obs, targets, test_size=0.2, random_state=42)
# train_obs_valid = np.ones((train_obs.shape[0], train_obs.shape[1], 1)).astype(bool)
# test_obs_valid = np.ones((test_obs.shape[0], test_obs.shape[1], 1)).astype(bool)
# Build Model
rkn = MVNXMotionPredictionRKN(observation_shape=train_obs.shape[-1], latent_observation_dim=args.latent_dim,
								 output_dim=train_targets.shape[-1], num_basis=15, bandwidth=3, never_invalid=True)
# rkn.load_weights(args.model_dir)
rkn = k.models.load_model(args.model_dir, custom_objects = {"RKN":RKN, "RKNTransitionCell": RKNTransitionCell, "rmse": rkn.rmse, "gaussian_nll":rkn.gaussian_nll})

test_preds = rkn.predict(test_obs)

np.save('rkn_mopred.npy', {"test_obs":test_obs, "test_preds":test_preds, "test_targets":test_targets})
