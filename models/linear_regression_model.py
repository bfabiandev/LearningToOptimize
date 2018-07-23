from models.base_model import BaseModel
import tensorflow as tf


class LinearRegressionModel(BaseModel):
    """Class containing the model of the optimizee - in this case a simple linear regressor with a single output,
    classifying the inputs into either 0 or 1.


    Arguments:
        config {dict} -- Configuration for model.
    """

    def __init__(self, config):
        super(LinearRegressionModel, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        # tf.set_random_seed(0)

        self.x = tf.placeholder(
            tf.float32, shape=[None, self.config["dim"]], name="input")
        self.y = tf.placeholder(tf.float32, shape=[None, 1], name="label")

        self.losses = []
        self.accuracy = None

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config["max_to_keep"])

    def create_problem(self, optimizer_name="rnn"):
        """Creates initial instance of the optimizee and returns the weights.

        Keyword Arguments:
            optimizer_name {str} -- Name that defines the optimizer, used to group the variables. (default: {"rnn"})

        Returns:
            weights {list} -- Returns weights in the initial problem.
        """

        with tf.variable_scope(optimizer_name):
            W = tf.get_variable(
                shape=[self.config["dim"], 1], initializer=tf.glorot_uniform_initializer(), name="kernel")
            b = tf.get_variable(
                shape=[1], initializer=tf.glorot_uniform_initializer(), name="bias")

            pred = tf.add(tf.matmul(self.x, W), b)

        self.losses.append(tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=pred)))

        return [W, b]

    def rollout(self, updates, weights, iteration):
        """Does one step of the rollout - creates new instance of the problem 
        where the weights are created by applying updates calculated by the optimizer.

        Arguments:
            updates {list} -- List of updates to be applied to the parameters.
            weights {list} -- List of parameters of the last layer of the rollout.

        Returns:
            weights {list} -- Returns ops to calculate the new weights.
        """

        weights = [var + update for update, var in zip(updates, weights)]
        pred = tf.add(tf.matmul(self.x, weights[0]), weights[1])

        self.losses.append(tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=pred)))

        if iteration == self.config["rollout_length"]:
            self.prediction = tf.round(tf.sigmoid(pred))
            correct_prediction = tf.equal(self.prediction, self.y)
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))

        return weights
