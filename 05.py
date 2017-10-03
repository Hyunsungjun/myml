# 2입력 1뉴런, 여러 데이터


# 2입력 2뉴런


# 2입력 2뉴런, 병합 층


# Lab 4 Multi-variable linear regression
import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1., 1], [2, 2], [3, 3]]
y_data = [[1.], [2], [3]]

W = tf.Variable(tf.random_normal([2, 1]))
b = tf.Variable(tf.random_normal([1]))
hypothesis = tf.matmul(x_data, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train])
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
