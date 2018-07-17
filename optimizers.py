import tensorflow as tf


class BaseOptimizer:
    def __init__(self, config):
        self.config = config

    def step(self, gradients, states):
        raise NotImplementedError


class SGD(BaseOptimizer):
    def __init__(self, config):
        super(SGD, self).__init__(config)
        self.learning_rate = config["learning_rate"]

    def step(self, gradients, states):
        return [(-self.learning_rate * grad, state) for grad, state in zip(gradients, states)]


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
        self.layers = config["layers"]
        self.hidden_size = config["hidden_size"]
        self.dim = config["dim"]
        self.saver = tf.train.Saver(max_to_keep=self.config["max_to_keep"])

        cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.LSTMCell(self.hidden_size) for _ in range(self.layers)])
        cell = tf.contrib.rnn.InputProjectionWrapper(cell, self.hidden_size)
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, 1)
        self.cell = tf.make_template('cell', cell)

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(0.0001)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.)
        return optimizer.apply_gradients(zip(gradients, v))

    def train(self, losses, sess, data, training_iters=4000, log_iters=1000):
        sum_losses = tf.reduce_sum(losses)

        apply_update = self.optimize(sum_losses)
        sess.run(tf.global_variables_initializer())

        ave = 0
        for i in range(training_iters):
            data_x, data_y = next(data.next_batch(self.config["batch_size"]))
    
            err, _ = sess.run([sum_losses, apply_update], feed_dict={"input:0" : data_x, "label:0" : data_y})
            ave += err
            if i % log_iters == 0:
                print(ave / log_iters if i != 0 else ave)
                ave = 0
        print(ave / log_iters)

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
        reshaped_grads = []
        for grad in gradients:
            reshaped_grads.append(tf.reshape(grad, shape=(-1, 1)))

        sizes = [tf.size(grad) for grad in reshaped_grads]
        reshaped_grads = tf.concat(reshaped_grads, 0)
        n_params = reshaped_grads.shape[0]

        # create inner state of RNN
        if states is None:
            states = [[tf.zeros([n_params, self.hidden_size])] * 2] * self.layers
        
        # define RNN op
        update, states = self.cell(reshaped_grads, states)
        
        # split and reshape inputs into their original shape
        updates = tf.split(update, sizes)
        updates = [tf.squeeze(update, axis=[1]) if tf.size(update) == 1 else update for update in updates]

        return updates, states
