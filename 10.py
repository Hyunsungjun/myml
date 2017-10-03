#logistic regression (classification)
import tensorflow as tf
from myplot import MyPlot

x = [[-2.], [-1], [1], [2]]
y = [[0.], [0], [1], [2]]

w = tf.Variable(tf.random_normal([1]))
hypo = tf.sigmoid(tf.matmul(x, w))

#cost = tf.reduce_mean((hypo - y) * (hypo - y))
cost = -tf.reduce_mean(y * tf.log(hypo) + (1 - y) * tf.log(1 - hypo))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

weights = []
errors = []

for i in range(1001):
    sess.run(train)

    if i % 20 == 0:
        weights.append(sess.run(w))
        errors.append(sess.run(cost))


