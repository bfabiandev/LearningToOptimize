import tensorflow as tf
from util import generate_data_sets, tsplot
import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_obj_func = 10
train_data_points = 1000
test_data_points = 1000
dim = 10
l2 = 0.0005
learning_rate = 0.01
momentum = 0.9
training_epochs = 1000
display_step = 100


def run_linear_regression_experiment(optimizers, data):
    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, dim])
    y = tf.placeholder(tf.float32, [None, 1])

    # Set model weights
    W = tf.get_variable("W", shape=[dim, 1], initializer=tf.glorot_normal_initializer())
    b = tf.get_variable("b", shape=[1], initializer=tf.zeros_initializer())

    # Construct model
    pred = tf.nn.sigmoid(tf.matmul(x, W) + b)

    # Minimize error using cross entropy
    unregularized_cost = tf.losses.sigmoid_cross_entropy(logits=pred, multi_class_labels=y)

    l2_loss = l2 * tf.nn.l2_loss(W)

    cost = tf.add(unregularized_cost, l2_loss, name='loss')

    # Test model
    correct_prediction = tf.equal(tf.round(pred), y)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    losses = []
    for optimizer_str in optimizers:
        opt = None

        if optimizer_str == 'gradient_descent':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            opt = optimizer.minimize(cost)
        elif optimizer_str == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            opt = optimizer.minimize(cost)
        elif optimizer_str == 'lbfgs':
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                cost, method='L-BFGS-B', options={'maxiter': 1})

        init = tf.global_variables_initializer()

        losses_opt = []
        for i, datum in enumerate(data):
            train_x, train_y, test_x, test_y = datum
            losses_current = []
            # Start training
            with tf.Session() as sess:

                # Run the initializer
                sess.run(init)

                # Training cycle
                for epoch in range(training_epochs):
                    avg_cost = 0.

                    if opt is not None:
                        # Run optimization op (backprop) and cost op (to get loss value)
                        _, c = sess.run([opt, cost], feed_dict={
                                        x: train_x, y: train_y})
                    else:
                        optimizer.minimize(sess, feed_dict={
                            x: train_x, y: train_y})
                        c = sess.run(cost, feed_dict={
                                     x: train_x, y: train_y})

                    #print(sess.run([W]))

                    # Compute average loss
                    avg_cost += c / len(train_x)

                    # Display logs per epoch step
                    if (epoch+1) % display_step == 0:
                        train_acc = accuracy.eval({x: train_x, y: train_y})
                        test_acc = accuracy.eval({x: test_x, y: test_y})

                        print("Epoch:", '%04d' % (epoch+1),
                              "cost = ", "{:.9f}".format(avg_cost),
                              "train acc = ", "{:.9f}".format(train_acc),
                              "test acc = ", "{:.9f}".format(test_acc))

                    losses_current.append(avg_cost)

            print("{}/{} optimization finished!".format(i+1, len(data)))

            losses_opt.append(losses_current)
        losses.append(losses_opt)
    return np.asarray(losses)


def main():
    data = generate_data_sets(
        dim, n_obj_func, train_data_points, test_data_points)

    optimizers = ['gradient_descent', 'momentum', 'lbfgs']

    lr_losses = run_linear_regression_experiment(optimizers, data)

    from scipy import stats
    print(stats.describe(lr_losses, axis=None))

    x = np.arange(training_epochs)

    ax = plt.subplot(111)
    for i in range(lr_losses.shape[0]):
        tsplot(ax, data=lr_losses[i])

    plt.show()


if __name__ == "__main__":
    main()
