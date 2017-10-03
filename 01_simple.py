# 1입력 1뉴런, 학습데이터 1개
import tensorflow as tf

x = [1]
y = [1]

#----- a neuron
w = tf.Variable(tf.random_normal([1]))
hypo = w * x
#-----

cost = (hypo - y) * (hypo - y)

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1001):
    sess.run(train)

    if i % 100 == 0:
        print(sess.run(w), sess.run(cost))

#how to predict
print(sess.run(hypo))

