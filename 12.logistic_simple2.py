import tensorflow as tf

x_data = [[-2.], [-1], [1], [2]]
y_data = [[0.], [0], [1], [1]]

#------- a neuron
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypo = tf.sigmoid(x_data * W + b)
#------- a neuron

cost = -tf.reduce_mean(y_data * tf.log(hypo) + tf.subtract(1., y_data) *
    tf.log(tf.subtract(1., hypo)))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    cost_val, _ = sess.run([cost, train])
    if step % 200 == 0:
        print(step, cost_val)

# Accuracy report
h = sess.run(hypo)
print("\nHypo: ", h)

predicted = tf.cast(hypo > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y_data), dtype=tf.float32))

# Accuracy report
h, c, a = sess.run([hypo, predicted, accuracy])
print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
