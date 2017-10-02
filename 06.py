# 1입력 1뉴런, 데이터 3개
import tensorflow as tf
from myplot import MyPlot

x = [[1.,1], [2,2], [3,3]] #(3, 2)
y = [[1.], [2], [3]] #(3, 1)

w = tf.Variable(tf.random_normal([2, 1])) #(2, 1)
b = tf.Variable(tf.random_normal([1]))
hypo = tf.matmul(x, w) + b  #(3, 1)

cost = tf.reduce_mean((hypo - y) * (hypo - y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

costs = []

for i in range(1001):
    sess.run(train)

    if i % 100 == 0:
        print(sess.run(w))
        print(sess.run(b))
        print(sess.run(cost))
        costs.append(sess.run(cost))

predict = tf.matmul([[4.,4]], w) + b  #새로 만들었다. 만약 위의 hypo를 그대로 사용하려면???
print(predict, sess.run(predict))



