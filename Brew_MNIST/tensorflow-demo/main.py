import tensorflow as tf

#for adding your own image:
import numpy as np
from PIL import Image



from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #y labels are oh-encoded

n_train = mnist.train.num_examples #55,000
n_validation = mnist.validation.num_examples #5000
n_test = mnist.test.num_examples #10,000

print(n_train)
print("Check if == 55000")
print(n_validation)
print("Check if == 5000")
print(n_test)
print("Check if == 10000")

n_input_layer = 784 # input layer (28 x 28 pixels)
n_1_layer = 512 #1st hidden layer
n_2_layer = 256 #2nd hidden layer
n_3_layer = 128 #3rd hidden layer
n_output = 10 #output layer (0-9 digits)

learning_rate = 0.0001 #1e-4, represents how much the parameters will adjust at each step of the learning process, larger learning rates can converge faster, but also have the potential to overshoot the optimal values as they are updated
n_iterations = 1000 #how many times we go through the training step
batch_size = 128 #refers to how many training examples we are using at each step
dropout = 0.5 #represents the threshold at which we eliminate some units at random

x = tf.placeholder("float", [None, n_input_layer])#shape == [None, 784] where None represents any amount as we are feeding in n number of images, each 784 pixels.
y = tf.placeholder("float", [None, n_output]) #none represents we are feeding in n number of photos and n_output 1 of 10 answers.

keep_prob = tf.placeholder(tf.float32) #used to control the dropout rate; initialize it as placeholder rather than variable.  Want to use it twice: first for training at 0.5 and second for testing at 1.0


##the parameters that the network will update in the training process are the "weight" and "bias" values.  We set an initial value rather than empty placeholder so the network can "learn" by modifying these values.
weights = {
    'w1' : tf.Variable(tf.truncated_normal([n_input_layer, n_1_layer], stddev = 0.1)),
    'w2' : tf.Variable(tf.truncated_normal([n_1_layer, n_2_layer], stddev = 0.1)),
    'w3' : tf.Variable(tf.truncated_normal([n_2_layer, n_3_layer], stddev = 0.1)),
    'out' : tf.Variable(tf.truncated_normal([n_3_layer, n_output], stddev = 0.1)),
}

biases = {
    'b1' : tf.Variable(tf.constant(0.1, shape = [n_1_layer])),
    'b2' : tf.Variable(tf.constant(0.1, shape = [n_2_layer])),
    'b3' : tf.Variable(tf.constant(0.1, shape = [n_3_layer])),
    'out' : tf.Variable(tf.constant(0.1, shape = [n_output]))
}

hidden_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1']) #matrix multiply previous layer, or in this case x input, by weights, then add biases.
hidden_2 = tf.add(tf.matmul(hidden_1, weights['w2']), biases['b2'])
hidden_3 = tf.add(tf.matmul(hidden_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(hidden_3, keep_prob)
output_layer = tf.matmul(hidden_3, weights['out']) + biases['out']




cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits = output_layer)
)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#in correct_pred, we use arg_max function to compare which images are being predicted correctly by looking at the output_layer (predictions) and y (labels).  Use the equal function to return this as a list of Booleans.

correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


#train on mini batches

for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={
             x: batch_x, y: batch_y, keep_prob: dropout
             })
#print loss and accuracy (per minibatch)
    if i % 100 == 0:
        minibatch_loss, minibatch_accuracy = sess.run(
                    [cross_entropy, accuracy], feed_dict = {x: batch_x, y:batch_y, keep_prob: 1.0}
        )
        print("Iteration", str(i), "\t| Loss =", str(minibatch_loss), "\t| Accuracy =", str(minibatch_accuracy))


test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
print("\nAccuracy on test set:", test_accuracy)

#to add an image
#img = np.invert(Image.open("test_img.png").convert('L')).ravel()

prediction = sess.run(tf.argmax(output_layer, 1), feed_dict={x: [img]})
print("preduction for test image:", np.squeeze(prediction))

#try to improve accuracy by playing with learning rate, batch size, iterations, and dropout rate.









