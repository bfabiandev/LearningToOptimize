import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data_loader.data_generator import MNISTDG, SimpleDG
from models.linear_regression_model import LinearRegressionModel
from models.mnist_model import MNISTModel
from optimizers import SGD, RMSprop, RNNOptimizer
from trainers.linear_regression_trainer import LinearRegressionTrainer
from utils.dirs import create_dirs
from utils.logger import Logger


def learn(optimizer, model, n_steps):
    """Given an optimizer, this function returns a list of losses, which can be evaluated in a session.

    Arguments:
        optimizer {[BaseOptimizer]} -- Model specifying the architecture of the optimizer.
        model {[BaseModel]} -- Model specifying the architecture of the optimizee.
        n_steps {[int]} -- Number of steps to roll out the optimization for.

    Returns:
        [list] -- List of tensors containing the losses. 
    """
    # create model
    x = model.create_problem(optimizer_name=optimizer.__class__.__name__)
    states = None
    for i in range(1, n_steps+1):
        # get loss
        loss = model.losses[i-1]
        # get gradients wrt loss
        grads = tf.gradients(loss, x)
        grads = [tf.stop_gradient(g) for g in grads]

        # create new model with updated weights
        updates, states = optimizer.step(grads, states)
        with tf.name_scope("rollout_{}".format(i)):
            x = model.rollout(weights=x, updates=updates, iteration=i)

    return model.losses


def main():

    config = {
        "optimizer": "rnn",
        "problem": "mnist",
        "rollout_length": 100,  # This is 100 in the paper
        "learning_rate": 0.1,
        "decay_rate": 0.9,
        "meta_layers": 2,
        "meta_hidden_size": 20,
        "layers": 2,
        "hidden_size": 100,
        "activation": 'relu',
        "preprocess": True,
        "max_to_keep": 3,
        "retrain": False,
        "dim": 10,
        "range_of_means": 10,
        "range_of_stds": 10,
        "summary_dir": "summary",
        "checkpoint_dir": "data_ckpt",
        "batch_size": 10000,
        "training_iters": 4000,
        "log_iters": 100
    }

    # create the experiments dirs
    create_dirs([config["summary_dir"], config["checkpoint_dir"]])
    # create tensorflow session
    sess = tf.Session()

    # create your data generator
    # create an instance of the model you want
    if config["problem"] == "simple":
        data = SimpleDG(config)
        model = LinearRegressionModel(config)
    elif config["problem"] == "mnist":
        data = MNISTDG(config)
        model = MNISTModel(config)
    else:
        raise ValueError("{} is not a valid problem".format(config["problem"]))

    # create tensorboard logger
    # logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    # trainer = LinearRegressionTrainer(sess, model, data, config, logger)

    sess.run(tf.global_variables_initializer())

    if config["optimizer"] == "sgd":
        optim = SGD(config)
        losses = learn(optim, model, config["rollout_length"])
    elif config["optimizer"] == "rms":
        optim = RMSprop(config)
        losses = learn(optim, model, config["rollout_length"])
    elif config["optimizer"] == "rnn":
        optim = RNNOptimizer(config)
        losses = learn(optim, model, config["rollout_length"])

        if config["retrain"]:
            optim.train(losses, sess, data)
        else:
            optim.load(sess)
    else:
        raise ValueError(
            "{} is not a valid optimizer".format(config["optimizer"]))

    # initialize variables in optimizee
    # (can't initialize all here because it would potentially overwrite the trained optimizer)
    sess.run(tf.variables_initializer(
        [var for var in tf.trainable_variables(scope=optim.__class__.__name__)]))

    x = np.arange(config["rollout_length"] + 1)

    for i in range(3):
        sess.run(tf.variables_initializer(
            [var for var in tf.trainable_variables(scope=optim.__class__.__name__)]))

        data.refresh_parameters(seed=i)
        data_x, data_y = next(data.next_batch(config["batch_size"]))

        l = sess.run([losses], feed_dict={
                     "input:0": data_x, "label:0": data_y})
        print(l)

        p1, = plt.semilogy(x, l[0], label=config["optimizer"])
        plt.legend(handles=[p1])
        plt.title('Losses')
        plt.show()

        # TODO compare different optimizers

    data.refresh_parameters()

    data_x, data_y = next(data.next_batch(100, mode="train"))
    pred = sess.run(model.prediction, feed_dict={
                    "input:0": data_x, "label:0": data_y})
    print(list(zip(pred, np.argmax(data_y, axis=1), pred == np.argmax(data_y, axis=1))))

    # calculate accuracy on test data
    seed = np.random.randint(low=0, high=1e6)
    data.refresh_parameters(seed=seed)
    data_x, data_y = next(data.next_batch(5000, mode="train"))
    acc = sess.run(model.accuracy, feed_dict={
                   "input:0": data_x, "label:0": data_y})
    print("Train accuracy: {}".format(acc))

    data_x, data_y = next(data.next_batch(5000, mode="test"))
    acc = sess.run(model.accuracy, feed_dict={
                   "input:0": data_x, "label:0": data_y})
    print("Test accuracy: {}".format(acc))


if __name__ == "__main__":
    main()
