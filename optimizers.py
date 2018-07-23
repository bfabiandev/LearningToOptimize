import os

import tensorflow as tf

from preprocess import preprocess_grads


class BaseOptimizer:
    def __init__(self, config):
        self.config = config

        self.init_global_step()
        self.saver = None

    def init_saver(self):
        raise NotImplementedError

    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(
                0, trainable=False, name='global_step')

    def save(self, sess):
        raise NotImplementedError

    def step(self, gradients, states):
        raise NotImplementedError


class SGD(BaseOptimizer):
    def __init__(self, config):
        super(SGD, self).__init__(config)
        self.learning_rate = config["learning_rate"]

    def step(self, gradients, states):
        if states is None:
            states = [None] * len(gradients)

        return [-self.learning_rate * grad for grad in gradients], states


class RMSprop(BaseOptimizer):
    def __init__(self, config):
        super(RMSprop, self).__init__(config)
        self.learning_rate = config["learning_rate"]
        self.decay_rate = config["decay_rate"]
        self.dim = config["dim"]

    def step(self, gradients, states):
        updates = []
        new_states = []

        if states is None:
            states = []
            for grad in gradients:
                states.append(tf.zeros_like(grad))
        for grad, state in zip(gradients, states):
            state = self.decay_rate*state + \
                (1-self.decay_rate)*tf.pow(grad, 2)
            update = -self.learning_rate*grad / (tf.sqrt(state)+1e-5)
            updates.append(update)
            new_states.append(state)
        return updates, new_states


class RNNOptimizer(BaseOptimizer):
    def __init__(self, config):
        super(RNNOptimizer, self).__init__(config)
        self.layers = config["meta_layers"]
        self.hidden_size = config["meta_hidden_size"]
        self.dim = config["dim"]

        cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.LSTMCell(self.hidden_size) for _ in range(self.layers)])
        cell = tf.contrib.rnn.InputProjectionWrapper(cell, self.hidden_size)
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, 1)
        self.cell = tf.make_template('cell', cell)

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(0.0001)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.)
        return optimizer.apply_gradients(zip(gradients, v), global_step=self.global_step_tensor)

    def train(self, losses, sess, data):
        sum_losses = tf.reduce_sum(losses)

        apply_update = self.optimize(sum_losses)
        sess.run(tf.global_variables_initializer())

        training_iters = self.config["training_iters"]
        log_iters = self.config["log_iters"]
        ave = 0
        for i in range(training_iters):
            data.refresh_parameters()

            data_x, data_y = next(data.next_batch(self.config["batch_size"]))

            err, _ = sess.run([sum_losses, apply_update], feed_dict={
                              "input:0": data_x, "label:0": data_y})

            ave += err
            if i % log_iters == 0:
                print("Loss after {} iterations: {}".format(
                    i, ave / log_iters if i != 0 else ave))
                ave = 0
        print("Loss after {} iterations: {}".format(i, ave / log_iters))
        self.save(sess)

    def step(self, gradients, states):
        """Do one step of the rollout.

        Arguments:
            gradients {list} -- List containing gradients (Tensors). 
            states {list or None} -- List containing inner state of RNN; or None if uninitialized.

        Returns:
            updates {list} -- List of updates for each parameter in the current rollout layer of the optimizee.
            states {list} -- State of RNN.
        """

        # flatten and concatenate inputs
        flattened_grads = []
        for grad in gradients:
            flattened_grads.append(tf.reshape(grad, shape=(-1, 1)))

        shapes = [tf.shape(grad) for grad in gradients]
        sizes = [tf.size(grad) for grad in flattened_grads]

        flattened_grads = tf.concat(flattened_grads, 0)

        if self.config["preprocess"]:
            flattened_grads = preprocess_grads(flattened_grads)

        n_params = flattened_grads.shape[0]

        # create inner state of RNN
        if states is None:
            states = [[tf.zeros([n_params, self.hidden_size])]
                      * 2] * self.layers

        # define RNN op
        with tf.variable_scope("meta"):
            update, states = self.cell(flattened_grads, states)

        # split and reshape inputs into their original shape
        updates = tf.split(update, sizes)
        updates = [tf.squeeze(update, axis=[1]) if tf.size(
            update) == 1 else update for update in updates]
        updates = [tf.reshape(update, shape)
                   for update, shape in zip(updates, shapes)]

        return updates, states

    def init_saver(self):
        self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="meta"),
                                    max_to_keep=self.config["max_to_keep"], )

    def save(self, sess):
        if self.saver is None:
            self.init_saver()

        print("Saving model...")
        self.saver.save(sess, os.path.join(
            os.getcwd(), self.config["checkpoint_dir"], "ckpt"), self.global_step_tensor)
        print("Model saved")

    def load(self, sess):
        if self.saver is None:
            self.init_saver()

        latest_checkpoint = tf.train.latest_checkpoint(
            os.path.join(os.getcwd(), self.config["checkpoint_dir"]))
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")
