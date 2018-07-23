from models.base_model import BaseModel
import tensorflow as tf
from utils.utils import get_nth_chunk


class MNISTModel(BaseModel):
    """Class containing the model of the optimizee - in this case a convolutional neural network with 10 outputs,
    classifying the inputs into 10 classes.

    Arguments:
        config {dict} -- Configuration for model.
    """

    def __init__(self, config):
        super(MNISTModel, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        # tf.set_random_seed(0)

        self.x = tf.placeholder(tf.float32, shape=[None, 784], name="input")
        self.y = tf.placeholder(tf.float32, shape=[None, 10], name="label")

        self.losses = []
        self.accuracy = None

        self.layers = self.config["layers"]
        self.hidden_size = self.config["hidden_size"]

        if hasattr(tf.nn, self.config["activation"]):
            self.activation_fn = getattr(tf.nn, self.config["activation"])
        else:
            raise ValueError("tf.nn does not have an activation called {}.".format(
                self.config["activation"]))

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config["max_to_keep"])

    def create_problem(self, optimizer_name="rnn"):
        """Creates initial instance of the optimizee and returns the weights.

        Keyword Arguments:
            optimizer_name {str} -- Name that defines the optimizer, used to group the variables. (default: {"rnn"})

        Returns:
            weights {list} -- Returns weights in the initial problem.
        """

        i = 0
        inputs = self.x
        labels = self.y

        with tf.variable_scope(optimizer_name):
            for layer in range(self.layers):
                with tf.variable_scope("layer_{}".format(layer)):
                    W = tf.get_variable(shape=[inputs.shape[-1], self.hidden_size],
                                        initializer=tf.glorot_uniform_initializer(), name="kernel")
                    b = tf.get_variable(
                        shape=[self.hidden_size], initializer=tf.glorot_uniform_initializer(), name="bias")

                    inputs = self.activation_fn(
                        tf.add(tf.matmul(inputs, W), b))

            with tf.variable_scope("layer_{}".format(self.layers)):
                W = tf.get_variable(shape=[
                                    inputs.shape[-1], 10], initializer=tf.glorot_uniform_initializer(), name="kernel")
                b = tf.get_variable(
                    shape=[10], initializer=tf.glorot_uniform_initializer(), name="bias")

                pred = tf.add(tf.matmul(inputs, W), b)

        weights = tf.trainable_variables()
        print(weights)
        self.losses.append(tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=pred)))

        return weights

    def rollout(self, updates, weights, iteration):
        """Does one step of the rollout - creates new instance of the problem 
        where the weights are created by applying updates calculated by the optimizer.

        Arguments:
            updates {list} -- List of updates to be applied to the parameters.
            weights {list} -- List of parameters of the last layer of the rollout.

        Returns:
            weights {list} -- Returns ops to calculate the new weights.
        """
        with tf.name_scope("update"):
            weights = [var + update for update, var in zip(updates, weights)]

        #chunk = get_nth_chunk(iteration, self.config["rollout_length"] + 1, tf.shape(self.x)[0])
        inputs = self.x
        labels = self.y

        for layer in range(self.layers):
            with tf.variable_scope("layer_{}".format(layer)):
                W = weights[2 * layer]
                b = weights[2 * layer + 1]

                inputs = self.activation_fn(tf.add(tf.matmul(inputs, W), b))

        with tf.variable_scope("layer_{}".format(self.layers)):
            W = weights[2 * self.layers]
            b = weights[2 * self.layers + 1]

            pred = tf.add(tf.matmul(inputs, W), b)

        self.losses.append(tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=pred)))

        if iteration == self.config["rollout_length"]:
            self.prediction = tf.argmax(pred, 1)
            correct_prediction = tf.equal(
                self.prediction, tf.argmax(labels, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))

        return weights
