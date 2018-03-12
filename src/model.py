import os
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import \
    InputLayer, Conv1d, MaxPool1d, \
    RNNLayer, DropoutLayer, DenseLayer, \
    LambdaLayer, ReshapeLayer, ConcatLayer, \
    Conv2d, MaxPool2d, FlattenLayer, DynamicRNNLayer, \
    DeConv2d, BatchNormLayer, ElementwiseLayer, \
    SubpixelConv2d, Seq2Seq, ExpandDimsLayer, TileLayer

import config

class Spacial_Model():

    def __init__(
            self,
            model_name,
            start_learning_rate,
            decay_rate,
            decay_steps
    ):
        self.start_learning_rate = start_learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.model_name = model_name

        self.__create_placeholders__()
        self.__create_model__()
        self.__create_loss__()
        self.__create_training_op__()

    def __create_placeholders__(self):
        self.x_root = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, config.in_seq_length, 1],
            name='input_x_root'
        )
        self.x_neighbour = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, config.in_seq_length, config.num_neighbour],
            name='input_x_neighbour'
        )
        self.decode_seqs = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, config.out_seq_length + 1, 1], # start_id at beginning
            name="decode_root_seqs"
        )
        self.decode_seqs_test = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, 1, 1], # start_id at beginning
            name="decode_root_seqs_test"
        )
        self.target_seqs = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, config.out_seq_length + 1, 1], # end_id at end
            name="target_root_seqs"
        )
        self.global_step = tf.placeholder(
            dtype=tf.int32,
            shape=[],
            name="global_step"
        )

    def __create_model__(self):
        self.train_net = self.__get_network__(
            self.x_root,
            self.x_neighbour,
            self.decode_seqs,
            is_train=True,
            reuse=False,
        )
        self.test_net = self.__get_network__(
            self.x_root,
            self.x_neighbour,
            self.decode_seqs_test,
            is_train=False,
            reuse=True,
        )
        self.train_net.print_params(False)
        self.train_net.print_layers()

    def __get_mape__(self, out, target):
        return tf.reduce_mean(tf.reduce_mean(tf.abs(out - target) / target, [1, 2]))

    def __create_loss__(self):
        self.mae_copy = tl.cost.absolute_difference_error(
            tf.slice(self.x_root, [0, config.in_seq_length - config.out_seq_length, 0], [config.batch_size, config.out_seq_length, 1]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            is_mean=True
        )
        # train loss
        self.nmse_train_loss = tl.cost.normalized_mean_square_error(self.train_net.outputs, self.target_seqs)
        self.nmse_train_noend = tl.cost.normalized_mean_square_error(
            tf.slice(self.train_net.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1])
        )
        self.mse_train_noend = tl.cost.mean_squared_error(
            tf.slice(self.train_net.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            is_mean=True
        )
        self.mae_train_noend = tl.cost.absolute_difference_error(
            tf.slice(self.train_net.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            is_mean=True
        )
        self.mape_train_noend = self.__get_mape__(
            tf.slice(self.train_net.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1])
        )
        # test loss
        self.nmse_test_loss = tl.cost.normalized_mean_square_error(self.test_net.outputs, self.target_seqs)
        self.nmse_test_noend = tl.cost.normalized_mean_square_error(
            tf.slice(self.test_net.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1])
        )
        self.mse_test_noend = tl.cost.mean_squared_error(
            tf.slice(self.test_net.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            is_mean=True
        )
        self.mae_test_noend = tl.cost.absolute_difference_error(
            tf.slice(self.test_net.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            is_mean=True
        )
        self.mape_test_noend = self.__get_mape__(
            tf.slice(self.test_net.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1])
        )
        # adaptive train loss
        self.train_loss = self.nmse_train_loss
        self.test_loss = self.nmse_test_loss

    def __create_training_op__(self):
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=self.start_learning_rate,
            global_step=self.global_step,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True,
            name="learning_rate"
        )
        all_vars = tl.layers.get_variables_with_name(self.model_name)
        self.optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
            .minimize(self.train_loss, var_list=all_vars)

    def __get_network__(self, encode_seq, neighbour_seq, decode_seq, is_train=True, reuse=False):
        w_init = tf.random_normal_initializer(stddev=0.02)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope(self.model_name, reuse=reuse) as vs:
            tl.layers.set_name_reuse(reuse)
            inputs_x_root = InputLayer(encode_seq, name='in_root')
            inputs_x_nbor = InputLayer(neighbour_seq, name="in_neighbour")

            # encoding neighbour graph information
            n = ReshapeLayer(inputs_x_nbor, (config.batch_size * config.in_seq_length, config.num_neighbour), "reshape1")
            n.outputs = tf.expand_dims(n.outputs, axis=-1)
            n = Conv1d(n, 4, 4, 1, act=tf.identity, padding='SAME', W_init=w_init, name='conv1')
            n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='bn1')
            n = MaxPool1d(n, 2, 2, padding='valid', name='maxpool1')
            n = FlattenLayer(n, name="flatten1")
            n = ReshapeLayer(n, (config.batch_size, config.in_seq_length, -1), name="reshape1_back")

            net_encode = ConcatLayer([inputs_x_root, n], concat_dim=-1, name="encode")
            net_decode = InputLayer(decode_seq, name="decode")

            net_rnn = Seq2Seq(
                net_encode, net_decode,
                cell_fn=tf.contrib.rnn.BasicLSTMCell,
                n_hidden=config.dim_hidden,
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length=tl.layers.retrieve_seq_length_op(net_encode.outputs),
                decode_sequence_length=tl.layers.retrieve_seq_length_op(net_decode.outputs),
                initial_state_encode=None,
                # dropout=(0.8 if is_train else None),
                dropout=None,
                n_layer=1,
                return_seq_2d=True,
                name='seq2seq'
            )
            # net_out = DenseLayer(net_rnn, n_units=64, act=tf.identity, name='dense1')
            net_out = DenseLayer(net_rnn, n_units=1, act=tf.identity, name='dense2')
            if is_train:
                net_out = ReshapeLayer(net_out, (config.batch_size, config.out_seq_length + 1, 1), name="reshape_out")
            else:
                net_out = ReshapeLayer(net_out, (config.batch_size, 1, 1), name="reshape_out")

            self.net_rnn = net_rnn

            return net_out

class Seq2Seq_Model(Spacial_Model):

    def __create_model__(self):
        self.train_net = self.__get_network__(
            self.x_root,
            self.decode_seqs,
            is_train=True,
            reuse=False,
        )
        self.test_net = self.__get_network__(
            self.x_root,
            self.decode_seqs_test,
            is_train=False,
            reuse=True,
        )
        self.train_net.print_params(False)
        self.train_net.print_layers()

    def __get_network__(self, encode_seq, decode_seq, is_train=True, reuse=False):
        w_init = tf.random_normal_initializer(stddev=0.02)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope(self.model_name, reuse=reuse) as vs:
            tl.layers.set_name_reuse(reuse)
            net_encode = InputLayer(encode_seq, name='in_root')

            net_decode = InputLayer(decode_seq, name="decode")

            net_rnn = Seq2Seq(
                net_encode, net_decode,
                cell_fn=tf.contrib.rnn.BasicLSTMCell,
                n_hidden=config.dim_hidden,
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length=tl.layers.retrieve_seq_length_op(net_encode.outputs),
                decode_sequence_length=tl.layers.retrieve_seq_length_op(net_decode.outputs),
                initial_state_encode=None,
                # dropout=(0.8 if is_train else None),
                dropout=None,
                n_layer=1,
                return_seq_2d=True,
                name='seq2seq'
            )
            # net_out = DenseLayer(net_rnn, n_units=64, act=tf.identity, name='dense1')
            net_out = DenseLayer(net_rnn, n_units=1, act=tf.identity, name='dense2')
            if is_train:
                net_out = ReshapeLayer(net_out, (config.batch_size, config.out_seq_length + 1, 1), name="reshape_out")
            else:
                net_out = ReshapeLayer(net_out, (config.batch_size, 1, 1), name="reshape_out")

            self.net_rnn = net_rnn

            return net_out

class WideDeep_Model(Spacial_Model):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.__create_placeholders_for_features__()
        super(WideDeep_Model, self).__init__(*args, **kwargs)

    def __create_placeholders_for_features__(self):

        self.features = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, config.out_seq_length + 1, config.dim_features],
            name='input_features'
        )

        self.features_test = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, 1, config.dim_features],
            name='input_features_test'
        )

    def __create_model__(self):
        self.train_net = self.__get_network__(
            self.x_root,
            self.decode_seqs,
            self.features,
            is_train=True,
            reuse=False,
        )
        self.test_net = self.__get_network__(
            self.x_root,
            self.decode_seqs_test,
            self.features_test,
            is_train=False,
            reuse=True,
        )
        self.train_net.print_params(False)
        self.train_net.print_layers()

    def __get_network__(self, encode_seq, decode_seq, features, is_train=True, reuse=False):
        w_init = tf.random_normal_initializer(stddev=0.02)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope(self.model_name, reuse=reuse) as vs:
            tl.layers.set_name_reuse(reuse)

            net_features = InputLayer(features, name="in_features")
            if is_train:
                net_features = ReshapeLayer(net_features, (config.batch_size * (config.out_seq_length + 1), config.dim_features), name="reshape_feature_1")
            else:
                net_features = ReshapeLayer(net_features, (config.batch_size * (1), config.dim_features), name="reshape_feature_1")

            net_features = DenseLayer(net_features, n_units=32, act=tf.nn.relu, name='dense_features')

            net_encode = InputLayer(encode_seq, name='in_root')
            net_decode = InputLayer(decode_seq, name="decode")

            net_rnn = Seq2Seq(
                net_encode, net_decode,
                cell_fn=tf.contrib.rnn.BasicLSTMCell,
                n_hidden=config.dim_hidden,
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length=tl.layers.retrieve_seq_length_op(net_encode.outputs),
                decode_sequence_length=tl.layers.retrieve_seq_length_op(net_decode.outputs),
                initial_state_encode=None,
                # dropout=(0.8 if is_train else None),
                dropout=None,
                n_layer=1,
                return_seq_2d=True,
                name='seq2seq'
            )

            # net_out = DenseLayer(net_rnn, n_units=64, act=tf.identity, name='dense1')
            net_out = ConcatLayer([net_rnn, net_features], concat_dim=-1, name="concat")
            net_out = DenseLayer(net_out, n_units=1, act=tf.identity, name='dense2')
            if is_train:
                net_out = ReshapeLayer(net_out, (config.batch_size, config.out_seq_length + 1, 1), name="reshape_out")
            else:
                net_out = ReshapeLayer(net_out, (config.batch_size, 1, 1), name="reshape_out")

            self.net_rnn = net_rnn

            return net_out

class Query_Model(Spacial_Model):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.__create_placeholders_for_query__()
        super(Query_Model, self).__init__(*args, **kwargs)

    def __create_placeholders_for_query__(self):

        self.query_x = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, config.in_seq_length, 1],
            name='input_query'
        )
        self.query_decode_seq = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, config.out_seq_length + 1, 1],
            name='decode_query'
        )
        self.query_decode_seq_test = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, 1, 1],
            name='decode_query'
        )

    def __create_model__(self):
        self.train_net = self.__get_network__(
            self.x_root,
            self.decode_seqs,
            self.query_decode_seq,
            is_train=True,
            reuse=False,
        )
        self.test_net = self.__get_network__(
            self.x_root,
            self.decode_seqs_test,
            self.query_decode_seq_test,
            is_train=False,
            reuse=True,
        )
        self.train_net.print_params(False)
        self.train_net.print_layers()

    def __get_network__(self, encode_seq, decode_seq, query_decode_seq, is_train=True, reuse=False):

        w_init = tf.random_normal_initializer(stddev=0.02)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope(self.model_name, reuse=reuse) as vs:
            tl.layers.set_name_reuse(reuse)
            net_encode_traffic = InputLayer(encode_seq, name='in_root_net')
            net_encode_query = InputLayer(self.query_x, name="in_query_net")
            net_encode = ConcatLayer([net_encode_traffic, net_encode_query], concat_dim=-1, name="encode")

            net_decode_traffic = InputLayer(decode_seq, name="decode_root")
            net_decode_query = InputLayer(query_decode_seq, name="decode_query_net")
            net_decode = ConcatLayer([net_decode_traffic, net_decode_query], concat_dim=-1, name="decode")

            net_rnn = Seq2Seq(
                net_encode, net_decode,
                cell_fn=tf.contrib.rnn.BasicLSTMCell,
                n_hidden=config.dim_hidden,
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length=tl.layers.retrieve_seq_length_op(net_encode.outputs),
                decode_sequence_length=tl.layers.retrieve_seq_length_op(net_decode.outputs),
                initial_state_encode=None,
                # dropout=(0.8 if is_train else None),
                dropout=None,
                n_layer=1,
                return_seq_2d=True,
                name='seq2seq'
            )
            # net_out = DenseLayer(net_rnn, n_units=64, act=tf.identity, name='dense1')
            net_out = DenseLayer(net_rnn, n_units=1, act=tf.identity, name='dense2')
            if is_train:
                net_out = ReshapeLayer(net_out, (config.batch_size, config.out_seq_length + 1, 1), name="reshape_out")
            else:
                net_out = ReshapeLayer(net_out, (config.batch_size, 1, 1), name="reshape_out")

            self.net_rnn = net_rnn

            return net_out

class Query_Comb_Model(Query_Model):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.__create_placeholders_for_query__()
        super(Query_Comb_Model, self).__init__(*args, **kwargs)

    def __create_placeholders_for_query__(self):

        self.query_x = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, config.in_seq_length, 1],
            name='input_query'
        )
        self.query_decode_seq = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, config.out_seq_length, 1],
            name='decode_query'
        )
        self.query_decode_seq_test = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, 1, 1],
            name='decode_query'
        )
        self.traffic_state = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size * (config.out_seq_length), config.dim_hidden],
            name="traffic_state"
        )

    def __create_training_op__(self):
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=self.start_learning_rate,
            global_step=self.global_step,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True,
            name="learning_rate"
        )
        all_model_vars = tl.layers.get_variables_with_name(self.model_name)
        self.optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
            .minimize(self.train_loss, var_list=all_model_vars)
        # self.optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
        #     .minimize(self.train_loss, var_list=all_vars)

    def __create_model__(self):
        self.train_seq2seq_rnn, self.train_seq2seq_out, self.train_query_rnn, self.train_net = self.__get_network__(
            self.x_root,
            self.decode_seqs,
            is_train=True,
            reuse=False,
        )
        self.test_seq2seq_rnn, self.test_seq2seq_out, self.test_query_rnn, self.test_net = self.__get_network__(
            self.x_root,
            self.decode_seqs_test,
            is_train=False,
            reuse=True,
        )
        self.train_net.print_params(False)
        self.train_net.print_layers()

    def __create_loss__(self):
        # train loss
        self.nmse_train_noend = tl.cost.normalized_mean_square_error(
            tf.slice(self.train_net.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1])
        )
        self.mape_train_noend = self.__get_mape__(
            tf.slice(self.train_net.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1])
        )
        self.train_loss = self.nmse_train_noend

    def __get_network__(self, encode_seq, decode_seq, is_train=True, reuse=False):
        w_init = tf.random_normal_initializer(stddev=0.02)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("seq2seq_model", reuse=reuse) as vs:
            tl.layers.set_name_reuse(reuse)
            net_encode = InputLayer(encode_seq, name='in_root')

            net_decode = InputLayer(decode_seq, name="decode")

            net_rnn = Seq2Seq(
                net_encode, net_decode,
                cell_fn=tf.contrib.rnn.BasicLSTMCell,
                n_hidden=config.dim_hidden,
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length=tl.layers.retrieve_seq_length_op(net_encode.outputs),
                decode_sequence_length=tl.layers.retrieve_seq_length_op(net_decode.outputs),
                initial_state_encode=None,
                # dropout=(0.8 if is_train else None),
                dropout=None,
                n_layer=1,
                return_seq_2d=True,
                name='seq2seq'
            )
            # self.net_rnn_seq2seq = net_rnn
            net_rnn_seq2seq = net_rnn

            net_out_seq2seq = DenseLayer(net_rnn, n_units=1, act=tf.identity, name='dense2')
            if is_train:
                net_out_seq2seq = ReshapeLayer(net_out_seq2seq, (config.batch_size, config.out_seq_length + 1, 1), name="reshape_out")
            else:
                net_out_seq2seq = ReshapeLayer(net_out_seq2seq, (config.batch_size, 1, 1), name="reshape_out")

            # net_out_seq2seq = net_out_seq2seq
            # net_out = DenseLayer(net_rnn, n_units=64, act=tf.identity, name='dense1')
            # net_out = DenseLayer(net_rnn, n_units=1, act=tf.identity, name='dense2')
            # net_out = ReshapeLayer(net_out, (config.batch_size, config.out_seq_length + 1, 1), name="reshape_out")

        with tf.variable_scope(self.model_name, reuse=reuse) as vs:
            tl.layers.set_name_reuse(reuse)
            net_encode_query = InputLayer(self.query_x, name='in_root_query')

            net_decode_query = InputLayer(self.query_decode_seq, name="decode_query")

            net_rnn_query = RNNLayer(
                net_decode_query,
                cell_fn=tf.contrib.rnn.BasicLSTMCell,
                cell_init_args={"forget_bias": 1.0},
                n_hidden=config.query_dim_hidden,
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                n_steps=config.out_seq_length,
                return_last=True,
                # return_last=False,
                # return_seq_2d=True,
                name="rnn_query"
            )
            net_rnn_query = ExpandDimsLayer(net_rnn_query, axis=1, name="rnn_query_expand")
            net_rnn_query = TileLayer(net_rnn_query, [1, config.out_seq_length, 1], name="rnn_query_tile")
            net_rnn_query = ReshapeLayer(net_rnn_query, (config.batch_size * config.out_seq_length, config.query_dim_hidden), name="rnn_query_reshape")

            net_traffic_state = InputLayer(self.traffic_state, name="in_traffic_state")

            if is_train:
                net_rnn_traffic = ReshapeLayer(net_rnn_seq2seq, (config.batch_size, config.out_seq_length + 1, config.dim_hidden), name="reshape_traffic_q1")
                net_rnn_traffic.outputs = tf.slice(net_rnn_traffic.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, config.dim_hidden], name="slice_traffic_q")
                net_rnn_traffic = ReshapeLayer(net_rnn_traffic, (config.batch_size * config.out_seq_length, config.dim_hidden), name="reshape_traffic_q2")
                net_out = ConcatLayer([net_rnn_traffic, net_rnn_query], concat_dim=-1, name="concat_traffic_query1")
            else:
                net_out = ConcatLayer([net_traffic_state, net_rnn_query], concat_dim=-1, name="concat_traffic_query2")

            # net_out = DenseLayer(net_out, n_units=128, act=tf.nn.relu, name="dense_query1")
            # net_out = DenseLayer(net_out, n_units=32, act=tf.nn.relu, name="dense_query2")
            net_out = DenseLayer(net_out, n_units=1, act=tf.identity, name="dense_query3")
            # net_out = ReshapeLayer(net_out, (config.batch_size, config.out_seq_length + 1, 1), name="reshape_out")
            # if is_train:
            net_out = ReshapeLayer(net_out, (config.batch_size, config.out_seq_length, 1), name="reshape_out")
            # else:
            #    net_out = ReshapeLayer(net_out, (config.batch_size, 1, 1), name="reshape_out")
        return net_rnn_seq2seq, net_out_seq2seq, net_rnn_query, net_out

class All_Comb_Model(Query_Comb_Model):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.__create_placeholders_for_features__()
        super(All_Comb_Model, self).__init__(*args, **kwargs)

    def __create_model__(self):
        self.train_net_seq2seq, self.train_net_spatial, self.train_net_wide, self.train_net_rnn_query, self.train_net_query = self.__get_network__(
            self.x_root,
            self.x_neighbour,
            self.decode_seqs,
            self.features,
            self.features,
            is_train=True,
            reuse=False,
        )
        self.test_net_seq2seq, self.test_net_spatial, self.test_net_wide, self.test_net_rnn_query, self.test_net_query = self.__get_network__(
            self.x_root,
            self.x_neighbour,
            self.decode_seqs_test,
            self.features_test,
            self.features,
            is_train=False,
            reuse=True,
        )
        self.train_net_query.print_params(False)
        self.train_net_query.print_layers()

    def __create_loss__(self):
        # train loss for spatial
        self.nmse_train_spatial = tl.cost.normalized_mean_square_error(
            tf.slice(self.train_net_spatial.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length + 1, 1]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length + 1, 1])
        )
        self.nmse_train_noend_spatial = tl.cost.normalized_mean_square_error(
            tf.slice(self.train_net_spatial.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1])
        )
        self.mape_train_noend_spatial = self.__get_mape__(
            tf.slice(self.train_net_spatial.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1])
        )
        self.train_loss_spatial = self.nmse_train_spatial
        # train loss for wide
        self.nmse_train_wide = tl.cost.normalized_mean_square_error(
            tf.slice(self.train_net_wide.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length + 1, 1]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length + 1, 1])
        )
        self.nmse_train_noend_wide = tl.cost.normalized_mean_square_error(
            tf.slice(self.train_net_wide.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1])
        )
        self.mape_train_noend_wide = self.__get_mape__(
            tf.slice(self.train_net_wide.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1])
        )
        self.train_loss_wide = self.nmse_train_wide
        # train loss for query
        self.nmse_train_noend_query = tl.cost.normalized_mean_square_error(
            tf.slice(self.train_net_query.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1])
        )
        self.mape_train_noend_query = self.__get_mape__(
            tf.slice(self.train_net_query.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1])
        )
        # self.train_loss_query = self.nmse_train_noend_query
        self.train_loss_query = self.mape_train_noend_query

    def __create_training_op__(self):
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=self.start_learning_rate,
            global_step=self.global_step,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True,
            name="learning_rate"
        )
        all_spatial_vars = tl.layers.get_variables_with_name(self.model_name + "_spatial")
        self.optim_spatial = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
            .minimize(self.train_loss_spatial, var_list=all_spatial_vars)
        all_wide_vars = tl.layers.get_variables_with_name(self.model_name + "_wide")
        self.optim_wide = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
            .minimize(self.train_loss_wide, var_list=all_wide_vars)
        all_query_vars = tl.layers.get_variables_with_name(self.model_name + "_query")
        self.optim_query = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
            .minimize(self.train_loss_query, var_list=all_query_vars)

    def __create_placeholders_for_features__(self):

        self.features = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, config.out_seq_length + 1, config.dim_features],
            name='input_features'
        )

        self.features_test = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, 1, config.dim_features],
            name='input_features_test'
        )

        self.base_pred = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, config.out_seq_length, 1],
            name='input_base_pred'
        )

    def __get_network__(self, encode_seq, neighbour_seq, decode_seq, features, features_full, is_train=True, reuse=False):
        w_init = tf.random_normal_initializer(stddev=0.02)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope(self.model_name + "_spatial", reuse=reuse) as vs:
            tl.layers.set_name_reuse(reuse)
            inputs_x_root = InputLayer(encode_seq, name='in_root')
            inputs_x_nbor = InputLayer(neighbour_seq, name="in_neighbour")

            # encoding neighbour graph information
            n = ReshapeLayer(inputs_x_nbor, (config.batch_size * config.in_seq_length, config.num_neighbour), "reshape1")
            n.outputs = tf.expand_dims(n.outputs, axis=-1)
            n = Conv1d(n, 4, 4, 1, act=tf.identity, padding='SAME', W_init=w_init, name='conv1')
            n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='bn1')
            n = MaxPool1d(n, 2, 2, padding='valid', name='maxpool1')
            n = FlattenLayer(n, name="flatten1")
            n = ReshapeLayer(n, (config.batch_size, config.in_seq_length, -1), name="reshape1_back")

            net_encode = ConcatLayer([inputs_x_root, n], concat_dim=-1, name="encode")
            net_decode = InputLayer(decode_seq, name="decode")

            net_rnn = Seq2Seq(
                net_encode, net_decode,
                cell_fn=tf.contrib.rnn.BasicLSTMCell,
                n_hidden=config.dim_hidden,
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length=tl.layers.retrieve_seq_length_op(net_encode.outputs),
                decode_sequence_length=tl.layers.retrieve_seq_length_op(net_decode.outputs),
                initial_state_encode=None,
                # dropout=(0.8 if is_train else None),
                dropout=None,
                n_layer=1,
                return_seq_2d=True,
                name='seq2seq'
            )
            net_rnn_seq2seq = net_rnn

            net_spatial_out = DenseLayer(net_rnn, n_units=1, act=tf.identity, name='dense2')
            if is_train:
                net_spatial_out = ReshapeLayer(net_spatial_out, (config.batch_size, config.out_seq_length + 1, 1), name="reshape_out")
            else:
                net_spatial_out = ReshapeLayer(net_spatial_out, (config.batch_size, 1, 1), name="reshape_out")

        with tf.variable_scope(self.model_name + "_wide", reuse=reuse) as vs:
            tl.layers.set_name_reuse(reuse)
            # Features
            net_features = InputLayer(features, name="in_features")
            net_features_full = InputLayer(features_full, name="in_features_full")
            net_features_full = ReshapeLayer(net_features_full, (config.batch_size * (config.out_seq_length + 1), config.dim_features), name="reshape_feature_full_1")
            if is_train:
                net_features = ReshapeLayer(net_features, (config.batch_size * (config.out_seq_length + 1), config.dim_features), name="reshape_feature_1")
            else:
                net_features = ReshapeLayer(net_features, (config.batch_size * (1), config.dim_features), name="reshape_feature_1")

            self.net_features_dim = 32
            net_features = DenseLayer(net_features, n_units=self.net_features_dim, act=tf.nn.relu, name='dense_features')
            net_features_full = DenseLayer(net_features_full, n_units=self.net_features_dim, act=tf.nn.relu, name='dense_features_full')
            # self.net_features = net_features

            net_wide_out = ConcatLayer([net_rnn_seq2seq, net_features], concat_dim=-1, name="concat_features")
            net_wide_out = DenseLayer(net_wide_out, n_units=1, act=tf.identity, name='dense2')

            if is_train:
                net_wide_out = ReshapeLayer(net_wide_out, (config.batch_size, config.out_seq_length + 1, 1), name="reshape_out")
            else:
                net_wide_out = ReshapeLayer(net_wide_out, (config.batch_size, 1, 1), name="reshape_out")

        with tf.variable_scope(self.model_name + "_query", reuse=reuse) as vs:
            tl.layers.set_name_reuse(reuse)

            net_decode_query = InputLayer(self.query_decode_seq, name="decode_query")

            net_rnn_query = RNNLayer(
                net_decode_query,
                cell_fn=tf.contrib.rnn.BasicLSTMCell,
                cell_init_args={"forget_bias": 1.0},
                n_hidden=config.query_dim_hidden,
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                n_steps=config.out_seq_length,
                return_last=True,

                # return_last=False,
                # return_seq_2d=True,
                name="rnn_query"
            )
            '''
            net_rnn_query = DynamicRNNLayer(
                net_decode_query,
                cell_fn=tf.contrib.rnn.BasicLSTMCell,
                cell_init_args={"forget_bias": 1.0},
                # n_hidden=config.query_dim_hidden,
                n_hidden=32,
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                return_last=True,
                # dropout=0.8,
                sequence_length=tl.layers.retrieve_seq_length_op(net_decode_query.outputs),
                # return_last=False,
                # return_seq_2d=True,
                name="rnn_query_dynamic"
            )
            '''

            net_rnn_query = ExpandDimsLayer(net_rnn_query, axis=1, name="rnn_query_expand")
            net_rnn_query = TileLayer(net_rnn_query, [1, config.out_seq_length, 1], name="rnn_query_tile")
            net_rnn_query = ReshapeLayer(net_rnn_query, (config.batch_size * config.out_seq_length, config.query_dim_hidden), name="rnn_query_reshape")
            # net_rnn_query = ReshapeLayer(net_rnn_query, (config.batch_size * config.out_seq_length, 32), name="rnn_query_reshape")

            # self.net_rnn_query = net_rnn_query

            net_traffic_state = InputLayer(self.traffic_state, name="in_traffic_state")

            '''
            if is_train:
                net_rnn_traffic = ReshapeLayer(net_rnn_seq2seq, (config.batch_size, config.out_seq_length + 1, config.dim_hidden), name="reshape_traffic_q1")
                net_rnn_traffic.outputs = tf.slice(net_rnn_traffic.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, config.dim_hidden], name="slice_traffic_q")
                net_rnn_traffic = ReshapeLayer(net_rnn_traffic, (config.batch_size * config.out_seq_length, config.dim_hidden), name="reshape_traffic_q2")

                net_features_traffic = ReshapeLayer(net_features, (config.batch_size, config.out_seq_length + 1, self.net_features_dim), name="reshape_features_q1")
                net_features_traffic.outputs = tf.slice(net_features_traffic.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, self.net_features_dim], name="slice_features_q")
                net_features_traffic = ReshapeLayer(net_features_traffic, (config.batch_size * config.out_seq_length, self.net_features_dim), name="reshape_features_q2")

                net_query_out = ConcatLayer([net_rnn_traffic, net_features_traffic, net_rnn_query], concat_dim=-1, name="concat_traffic_query1")
                # net_query_out = ConcatLayer([net_rnn_traffic, net_rnn_query], concat_dim=-1, name="concat_traffic_query1")
            else:
            '''
            net_features_traffic = ReshapeLayer(net_features_full, (config.batch_size, config.out_seq_length + 1, self.net_features_dim), name="reshape_features_q1")
            net_features_traffic.outputs = tf.slice(net_features_traffic.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, self.net_features_dim], name="slice_features_q")
            net_features_traffic = ReshapeLayer(net_features_traffic, (config.batch_size * config.out_seq_length, self.net_features_dim), name="reshape_features_q2")

            net_query_out = ConcatLayer([net_traffic_state, net_features_traffic, net_rnn_query], concat_dim=-1, name="concat_traffic_query1")
                # net_rnn_traffic = ReshapeLayer(net_rnn_seq2seq, (config.batch_size, config.out_seq_length + 1, config.dim_hidden), name="reshape_traffic_q1")
                # net_rnn_traffic.outputs = tf.slice(net_rnn_traffic.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, config.dim_hidden], name="slice_traffic_q")
                # net_rnn_traffic = ReshapeLayer(net_rnn_traffic, (config.batch_size * config.out_seq_length, config.dim_hidden), name="reshape_traffic_q2")
                # net_query_out = ConcatLayer([net_rnn_traffic, net_features_traffic, net_rnn_query], concat_dim=-1, name="concat_traffic_query1")

            # net_out = DenseLayer(net_out, n_units=128, act=tf.nn.relu, name="dense_query1")
            # net_out = DenseLayer(net_out, n_units=64, act=tf.nn.relu, name="dense_query2")
            # net_query_out = DropoutLayer(net_query_out, keep=0.8, is_fix=True, is_train=is_train, name='drop_query3')
            net_query_out = DenseLayer(net_query_out, n_units=1, act=tf.identity, name="dense_query3")
            # net_out = ReshapeLayer(net_out, (config.batch_size, config.out_seq_length + 1, 1), name="reshape_out")
            # if is_train:
            net_query_out = ReshapeLayer(net_query_out, (config.batch_size, config.out_seq_length, 1), name="reshape_out")
            # else:
            #    net_out = ReshapeLayer(net_out, (config.batch_size, 1, 1), name="reshape_out")

            # TODO residual net
            '''
            if is_train:
                net_query_out.outputs = tf.add(
                    net_query_out.outputs,
                    tf.slice(net_wide_out.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, 1]),
                    name="res_add"
                )
            else:
            '''
            net_base_pred = InputLayer(self.base_pred, name="in_net_base_pred")
            net_query_out.outputs = tf.add(net_query_out.outputs, net_base_pred.outputs, name="res_add")

        return net_rnn_seq2seq, net_spatial_out, net_wide_out, net_rnn_query, net_query_out

if __name__ == "__main__":
    '''
    model = Spacial_Model(
        model_name="spacial_model",
        start_learning_rate=0.001,
        decay_steps=400,
        decay_rate=0.8,
    )
    '''
    '''
    model = Seq2Seq_Model(
        model_name="seq2seq_model",
        start_learning_rate=0.001,
        decay_steps=400,
        decay_rate=0.8,
    )
    '''
    '''
    model = WideDeep_Model(
        model_name="widedeep_model",
        start_learning_rate=0.001,
        decay_steps=400,
        decay_rate=0.8,
    )
    '''
    model = Query_Comb_Model(
        model_name="query_comb_model_%d_%d" % (config.impact_k, config.query_dim_hidden) if config.query_dim_hidden != config.dim_hidden else "query_comb_model_%d" % (config.impact_k),
        start_learning_rate=0.001,
        decay_steps=400,
        decay_rate=0.8
    )
    '''
    model = All_Comb_Model(
        model_name="all_comb_model_%d_%d" % (config.impact_k, config.query_dim_hidden) if config.query_dim_hidden != config.dim_hidden else "all_comb_model_%d" % (config.impact_k),
        # model_name="all_comb_model_%d" % config.impact_k,
        start_learning_rate=0.001,
        decay_steps=1e4,
        decay_rate=0.8
    )
    '''
