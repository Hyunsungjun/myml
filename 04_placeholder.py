# 1입력 1뉴런, 학습데이터 1개, 플레이스 홀더
# (1)1b-1/R/PH
import tensorflow as tf

x_data = [1]
y_data = [1]

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

#----- a neuron
w = tf.Variable(tf.random_normal([1]))
hypo = w * x
#-----

cost = (hypo - y) * (hypo - y)

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1001):
    sess.run(train, feed_dict={x:x_data, y:y_data})

    if i % 50 == 0:
        print(sess.run(w), sess.run(cost, feed_dict={x:x_data, y:y_data}))

print(sess.run(hypo, feed_dict={x:[5]}))
