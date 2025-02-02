import tensorflow as tf
from tensorflow import keras as k
import numpy as np
from rkn.RKNTransitionCell import RKNTransitionCell, pack_input, unpack_state


class RKN(k.models.Model):

    def __init__(self, observation_shape, latent_observation_dim, output_dim, num_basis, bandwidth,
                 trans_net_hidden_units=[], never_invalid=False, cell_type="rkn", batch_size=None):
        """
        :param observation_shape: shape of the observation to work with
        :param latent_observation_dim: latent observation dimension (m in paper)
        :param output_dim: dimensionality of model output
        :param num_basis: number of basis matrices (k in paper)
        :param bandwidth: bandwidth of transition sub-matrices (b in paper)
        :param trans_net_hidden_units: hidden units for transition network
        :param never_invalid: if you know a-priori that the observation valid flag will always be positive you can set
                              this to true for slightly increased performance (obs_valid mask will be ignored)
        :param cell_type: type of cell to use "rkn" for our approach, "lstm" or "gru" for baselines
        :param batch_size: For stateful RNNs, batch size needs to be fixed.
        """
        super().__init__()

        self._obs_shape = observation_shape
        self._lod = latent_observation_dim
        self._lsd = 2 * self._lod
        self._output_dim = output_dim
        self._never_invalid = never_invalid
        self._ld_output = np.isscalar(self._output_dim)

        # build encoder
        self._enc_hidden_layers = self._time_distribute_layers(self.build_encoder_hidden())

        # we need to ensure the bias is initialized with non-zero values to ensure the normalization does not produce
        # nan
        self._layer_w_mean = k.layers.TimeDistributed(
            k.layers.Dense(self._lod, activation=k.activations.linear,
                           bias_initializer=k.initializers.RandomNormal(stddev=0.05)))
        self._layer_w_mean_norm = k.layers.TimeDistributed(k.layers.Lambda(
            lambda x: x / tf.norm(x, ord='euclidean', axis=-1, keepdims=True)))
        self._layer_w_covar = k.layers.TimeDistributed(
            k.layers.Dense(self._lod, activation=lambda x: k.activations.elu(x) + 1))

        # build transition
        if cell_type.lower() == "rkn":
            self._cell = RKNTransitionCell(self._lsd, self._lod,
                                           number_of_basis=num_basis,
                                           bandwidth=bandwidth,
                                           trans_net_hidden_units=trans_net_hidden_units,
                                           never_invalid=never_invalid)
        elif cell_type.lower() == "lstm":
            print("Running LSTM Baseline")
            self._cell = k.layers.LSTMCell(2 * self._lsd)
        elif cell_type.lower() == "gru":
            print("Running GRU Baseline")
            self._cell = k.layers.GRUCell(2 * self._lsd)
        else:
            raise AssertionError("Invalid Cell type, needs tp be 'rkn', 'lstm' or 'gru'")

        self._layer_rkn = k.layers.RNN(self._cell, return_sequences=True, stateful = batch_size is not None)

        self._dec_hidden = self._time_distribute_layers(self.build_decoder_hidden())
        if self._ld_output:
            # build decoder mean
            self._layer_dec_out = k.layers.TimeDistributed(k.layers.Dense(units=self._output_dim))

            # build decoder variance
            self._var_dec_hidden = self._time_distribute_layers(self.build_var_decoder_hidden())
            self._layer_var_dec_out = k.layers.TimeDistributed(
                k.layers.Dense(units=self._output_dim, activation=lambda x: k.activations.elu(x) + 1))

        else:
            self._layer_dec_out = k.layers.TimeDistributed(
                k.layers.Conv2DTranspose(self._output_dim[-1], kernel_size=3, padding="same",
                                         activation=k.activations.sigmoid))

        # To ensure the time sequence length remains unknown and the model can be used with varying sequence lengths
        if isinstance(observation_shape, list):
            in_shape = [None] + observation_shape
        elif isinstance(observation_shape, tuple):
            in_shape = list((None, ) + observation_shape)
        elif np.isscalar(observation_shape):
            in_shape = [None, observation_shape]
        else:
            raise AssertionError("observation shape needs to be either list, tuple or scalar")
        if batch_size is None:
            inputs = k.layers.Input(shape=in_shape)
        else:
            inputs = k.layers.Input(batch_shape = [batch_size] + in_shape)
        if self._never_invalid:
            self(inputs)
        else:
            if batch_size is None:
                obs_val = k.layers.Input(shape=(None, 1))
            else:
                obs_val = k.layers.Input(batch_shape = (batch_size, None, 1))
            self((inputs, obs_val))

    def build_encoder_hidden(self):
        """
        Implement encoder hidden layers
        :return: list of encoder hidden layers
        """
        raise NotImplementedError

    def build_decoder_hidden(self):
        """
        Implement mean decoder hidden layers
        :return: list of mean decoder hidden layers
        """
        raise NotImplementedError

    def build_var_decoder_hidden(self):
        """
        Implement var decoder hidden layers
        :return: list of var decoder hidden layers
        """
        raise NotImplementedError

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: model inputs (i.e. observations)
        :param training: required by k.models.Models
        :param mask: required by k.models.Model
        :return:
        """
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            img_inputs, obs_valid = inputs
        else:
            assert self._never_invalid, "If invalid inputs are possible, obs_valid mask needs to be provided"
            img_inputs = inputs
            obs_valid = tf.ones([tf.shape(img_inputs)[0], tf.shape(img_inputs)[1], 1])
        img_inputs = tf.cast(img_inputs, tf.float32)

        # encoder
        enc_last_hidden = self._prop_through_layers(img_inputs, self._enc_hidden_layers)
        w_mean = self._layer_w_mean_norm(self._layer_w_mean(enc_last_hidden))
        w_covar = self._layer_w_covar(enc_last_hidden)

        # transition
        rkn_in = pack_input(w_mean, w_covar, obs_valid)
        z = self._layer_rkn(rkn_in)

        post_mean, post_covar = unpack_state(z)
        post_covar = tf.concat(post_covar, -1)

        # decode
        pred_mean = self._layer_dec_out(self._prop_through_layers(post_mean, self._dec_hidden))
        if self._ld_output:
            pred_var = self._layer_var_dec_out(self._prop_through_layers(post_covar, self._var_dec_hidden))
            return tf.concat([pred_mean, pred_var], -1)
        else:
            return pred_mean

    # loss functions

    @staticmethod
    def gaussian_nll(target, pred_mean_var):
        """
        gaussian nll
        :param target: ground truth positions
        :param pred_mean_var: mean and covar (as concatenated vector, as provided by model)
        :return: gaussian negative log-likelihood
        """
        pred_dim = pred_mean_var.shape[-1]//2
        pred_mean, pred_var = pred_mean_var[..., :pred_dim], pred_mean_var[..., pred_dim:]
        pred_var += 1e-8
        element_wise_nll = 0.5 * (np.log(2 * np.pi) + tf.math.log(pred_var) + ((target - pred_mean)**2) / pred_var)
        sample_wise_error = tf.reduce_sum(element_wise_nll, axis=-1)
        return tf.reduce_sum(sample_wise_error)

    @staticmethod
    def rmse(target, pred_mean_var):
        """
        root mean squared error
        :param target: ground truth positions
        :param pred_mean_var: mean and covar (as concatenated vector, as provided by model)
        :return: root mean squared error between targets and predicted mean, predicted variance is ignored
        """
        pred_dim = pred_mean_var.shape[-1]//2
        pred_mean = pred_mean_var[..., :pred_dim]
        return tf.sqrt(tf.reduce_sum((pred_mean - target) ** 2))

    @staticmethod
    def bernoulli_nll(targets, predictions, uint8_targets=False):
        """ Computes Binary Cross Entropy
        :param targets:
        :param predictions:
        :param uint8_targets: if true it is assumed that the targets are given in uint8 (i.e. the values are integers
        between 0 and 255), thus they are devided by 255 to get "float image representation"
        :return: Binary Crossentropy between targets and prediction
        """
        if uint8_targets:
            targets = tf.cast(targets,tf.float32) / 255.0
        point_wise_error = - (
                    targets * tf.math.log(predictions + 1e-12) + (1 - targets) * tf.math.log(1 - predictions + 1e-12))
        red_axis = [i + 2 for i in range(len(targets.get_shape()) - 2)]
        sample_wise_error = tf.reduce_sum(point_wise_error, axis=red_axis)
        return tf.reduce_mean(sample_wise_error)

    # helpers
    @staticmethod
    def _prop_through_layers(inputs, layers):
        """propagates inputs through layers"""
        h = inputs
        for layer in layers:
            h = layer(h)
        return h

    @staticmethod
    def _time_distribute_layers(layers):
        """wraps layers with k.layers.TimeDistributed"""
        td_layers = []
        for l in layers:
            td_layers.append(k.layers.TimeDistributed(l))
        return td_layers
