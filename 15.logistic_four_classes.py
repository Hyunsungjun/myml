import tensorflow as tf

x_data = [[0., 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0.], [1], [1], [0]]

#------- two neurons - one neuron
W1 = tf.Variable(tf.random_normal([2, 2]))
b1 = tf.Variable(tf.random_normal([2]))
o1 = tf.sigmoid(tf.matmul(x_data, W1) + b1)

W2 = tf.Variable(tf.random_normal([2, 1]))
b2 = tf.Variable(tf.random_normal([1]))
hypo = tf.sigmoid(tf.matmul(o1, W2) + b2)

#----- learning
cost = -tf.reduce_mean(y_data * tf.log(hypo) + tf.subtract(1., y_data) *
    tf.log(tf.subtract(1., hypo)))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20001):
    cost_val, _ = sess.run([cost, train])
    if step % 200 == 0:
        print(step, cost_val)

#----- testing(classification)
predicted = tf.cast(hypo > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y_data), dtype=tf.float32))

h = sess.run(hypo)
print("\nHypo: ", h)

p = sess.run(predicted)
print("Predicted: ", p)

a = sess.run(accuracy)
print("Accuracy(%): ", a * 100)
