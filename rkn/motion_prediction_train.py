import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from sklearn.model_selection import train_test_split
import sys

import argparse

from rkn.RKN import RKN
from util.LayerNormalization import LayerNormalization
from util.MVNXDataDriver import *


parser = argparse.ArgumentParser(description='Train Autoencoder network')
parser.add_argument('--batch-size', type=int, default=24, metavar='B',
                    help='Input batch size for training (default: 24).')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='LR',
                    help='Learning rate for the optimizer (default: 1e-3).')
parser.add_argument('--epochs', type=int, default=100, metavar='E',
                    help='Number of epochs to train (default: 100).')
parser.add_argument('--hidden-dim', type=int, default=50, metavar='H',
                    help='Dimensions of the hidden layers (default: 50).')
parser.add_argument('--latent-dim', type=int, default=5, metavar='L',
                    help='Dimensions of the latent space (default: 5).')
parser.add_argument('--seed', type=int, default=89514, metavar='S',
                    help='Random seed (default: 89514).')
parser.add_argument('--src-dir', type=str, metavar='SRC', required=True,
                    help='Directory to the npy files where the skeletons are stored.')
parser.add_argument('--model-dir', type=str, metavar='DIR', required=True,
                    help='Directory where the model files should be saved to.')
parser.add_argument('--checkpoint', type=str, default=None, metavar='CKPT',
                    help='Path to saved checkpoint from which training should resume. (default: None)')
parser.add_argument('--seq-len', type=int, default=540, metavar='LEN',
                    help='Sequences length (default: 540)')
parser.add_argument('--test-split', type=float, default=0.2, metavar='S',
                    help='What fraction of the data should be used for testing (default: 0.2)')
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
train_obs, test_obs, train_targets, test_targets = train_test_split(obs, targets, test_size=args.test_split, random_state=42)
# train_obs_valid = np.ones((train_obs.shape[0], train_obs.shape[1], 1)).astype(bool)
# test_obs_valid = np.ones((test_obs.shape[0], test_obs.shape[1], 1)).astype(bool)
# Build Model
rkn = MVNXMotionPredictionRKN(observation_shape=train_obs.shape[-1], latent_observation_dim=args.latent_dim,
								 output_dim=train_targets.shape[-1], num_basis=15, bandwidth=3, never_invalid=True)
rkn.compile(optimizer=k.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=5.0), loss=rkn.rmse, metrics=[rkn.rmse])

# Train Model
rkn.fit(train_obs, train_targets, batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(test_obs, test_targets))
	
rkn.save(args.model_dir)