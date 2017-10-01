# 1입력 1뉴런, 데이터 1개

import tensorflow as tf
import lib.myplot as myplot

x = [1]
y = [1]

w = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))
hypo = w * x + b

cost = (hypo - y) * (hypo - y)

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost_fun)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

weights = []
biases = []
costs = []

for i in range(1001):
    sess.run(train)
    if i % 50 == 0:
        weights.append(sess.run(w))
        biases.append(sess.run(b))
        costs.append(sess.run(cost_fun))


gildong = pl.MyPlot()
gildong.show_list(costs)
