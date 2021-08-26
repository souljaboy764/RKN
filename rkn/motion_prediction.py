import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from sklearn.model_selection import train_test_split
import sys

from rkn.RKN import RKN
from util.LayerNormalization import LayerNormalization
from util.MVNXDataDriver import *

def generate_motion_prediction_set(root_dir, seq_length):
	DATA_PARAMS = {}
	DATA_PARAMS.update({"data_source": "MVNX", "nb_frames":seq_length, 'as_3D': True, 'data_types': ['position'],"unit_bounds": False, "path":root_dir})
	data_driver = MVNXDataDriver(DATA_PARAMS)
	data_driver.parse(frameMod=True)
	shape = data_driver.data.shape
	data = data_driver.data.reshape(-1, shape[-2], shape[-1])
	return data, data.copy()


# Implement Encoder and Decoder hidden layers
class MVNXMotionPredictionRKN(RKN):

	def build_encoder_hidden(self):
		return [
			# 1: Dense Layer (Input to hidden)
			k.layers.Dense(50, activation=k.activations.relu),
			# 2: Dense Layer (hidden to latent)
			k.layers.Dense(2*self._lod, activation=k.activations.relu)
		]

	def build_decoder_hidden(self):
		return [
			# 1: Dense Layer (latent to hidden)
			k.layers.Dense(50, activation=k.activations.relu),
			# 2: Dense Layer (hidden to Input)
			k.layers.Dense(self._output_dim, activation=k.activations.relu)
		]
	
	def build_var_decoder_hidden(self):
		return [
			# 1: Dense Layer (latent to hidden)
			k.layers.Dense(50, activation=k.activations.relu),
			# 2: Dense Layer (hidden to Input)
			k.layers.Dense(self._output_dim, activation=k.activations.relu)
		]


# Read Data
if len(sys.argv)<2:
	print('Usage: python3 motion_prediction.py /path/to/xml/files')
	exit()
obs, targets = generate_motion_prediction_set(sys.argv[1], 540)
train_obs, test_obs, train_targets, test_targets = train_test_split(obs, targets, test_size=0.2, random_state=42)
train_obs_valid = np.ones((train_obs.shape[0], train_obs.shape[1], 1)).astype(bool)
test_obs_valid = np.ones((test_obs.shape[0], test_obs.shape[1], 1)).astype(bool)
# Build Model
rkn = MVNXMotionPredictionRKN(observation_shape=train_obs.shape[-1], latent_observation_dim=5,
								 output_dim=train_targets.shape[-1], num_basis=15, bandwidth=3, never_invalid=False)
rkn.compile(optimizer=k.optimizers.Adam(clipnorm=5.0), loss=rkn.gaussian_nll, metrics=[rkn.rmse])

# Train Model
rkn.fit((train_obs, train_obs_valid), train_targets, batch_size=10, epochs=50,
        validation_data=((test_obs, test_obs_valid), test_targets))