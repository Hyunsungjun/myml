# 1입력 1뉴런, 데이터 1개
import tensorflow as tf

x = [1]
y = [1]

w = tf.Variable(tf.random_normal([1]))
hypo = w * x

cost = (hypo - y) * (hypo - y)

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

costs = []

for i in range(1001):
    sess.run(train)
    if i % 50 == 0:
        print(sess.run(w), sess.run(cost))
        costs.append(sess.run(cost))

