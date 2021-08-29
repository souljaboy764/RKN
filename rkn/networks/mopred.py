from tensorflow import keras as k
from sklearn.model_selection import train_test_split

from rkn.RKN import RKN

# Implement Encoder and Decoder hidden layers
class MVNXMotionPredictionRKN(RKN):

	def __init__(self, observation_shape, latent_observation_dim, hidden_layer_dim, output_dim, num_basis, bandwidth, trans_net_hidden_units=[], never_invalid=False, cell_type="rkn", batch_size=None):
		self._hidden_dim = hidden_layer_dim
		super().__init__(observation_shape, latent_observation_dim, output_dim, num_basis, bandwidth, trans_net_hidden_units=trans_net_hidden_units, never_invalid=never_invalid, cell_type=cell_type, batch_size=batch_size)

	def build_encoder_hidden(self):
		return [
			# 1: Dense Layer (Input to hidden)
			k.layers.Dense(self._hidden_dim, activation=k.activations.relu),
			k.layers.Dense(self._hidden_dim, activation=k.activations.relu),
			# 2: Dense Layer (hidden to latent)
			k.layers.Dense(self._lsd, activation=k.activations.relu)
		]

	def build_decoder_hidden(self):
		return [
			# 1: Dense Layer (latent to hidden)
			k.layers.Dense(self._hidden_dim, activation=k.activations.relu),
			k.layers.Dense(self._hidden_dim, activation=k.activations.relu),
			# 2: Dense Layer (hidden to Input)
			k.layers.Dense(self._output_dim, activation=k.activations.relu)
		]
	
	def build_var_decoder_hidden(self):
		return [
			# 1: Dense Layer (latent to hidden)
			k.layers.Dense(self._hidden_dim, activation=k.activations.relu),
			k.layers.Dense(self._hidden_dim, activation=k.activations.relu),
			# 2: Dense Layer (hidden to Input)
			k.layers.Dense(self._output_dim, activation=k.activations.relu)
		]