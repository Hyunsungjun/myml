# 1입력 1뉴런, 데이터 3개

import tensorflow as tf
import lib.myplot as myplot

x = [1, 2, 3]
y = [1, 2, 3]

w = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))
hypo = w * x + b

cost = tf.reduce_mean((hypo - y) * (hypo - y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

weights = []
biases = []
errors = []

for i in range(1001):
    sess.run(train)

    if i % 20 == 0:
        weights.append(sess.run(w))
        biases.append(sess.run(b))
        errors.append(sess.run(cost))

p = myplot.MyPlot()
p.show_list(errors)




