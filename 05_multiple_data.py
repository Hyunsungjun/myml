# (3)1-1/R
import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

#----- a neuron
w = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))
hypo = x_data*w
# 1*w       1
# 2*w       2   (w)
# 3*2       3

#----- learning
cost = tf.reduce_mean((hypo - y_data) ** 2)

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_list = []
print(sess.run(w), sess.run(b), sess.run(cost))
for i in range(1001):
    sess.run(train)

    if i % 50 == 0:
        print(sess.run(w), sess.run(b), sess.run(cost))
        cost_list.append(sess.run(cost))

# Show the error
import matplotlib.pyplot as plt
plt.plot(cost_list)
plt.show();

#----- testing(prediction)
print(hypo) #shape=(3,)
print(sess.run(hypo))


