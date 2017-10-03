# 1입력 1뉴런, 학습데이터 1개
import tensorflow as tf

x = [1]
y = [1]

x_p = tf.placeholder(tf.float32)
y_p = tf.placeholder(tf.float32)

#----- a neuron
w = tf.Variable(tf.random_normal([1]))
hypo = w * x_p

#----- learning
cost = (hypo - y_p) * (hypo - y_p)

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1001):
    sess.run(train, feed_dict={x_p:x, y_p:y})

    if i % 100 == 0:
        print(sess.run(w), sess.run(cost, feed_dict={x_p:x, y_p:y}))

#----- testing(prediction)
print(sess.run(hypo, feed_dict={x_p:[5]}))

