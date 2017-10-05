

# 2입력 1뉴런, 여러 데이터


# 2입력 2뉴런


# 2입력 2뉴런, 병합 층



# Lab 4 Multi-variable linear regression
import tensorflow as tf
a1 = tf.Variable([[0.1, 0.7, 0.5],
                  [0.2, 0.3, 0.6]])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(tf.argmax(a1, 1)))