# (1)1-1/R
import tensorflow as tf

<<<<<<< HEAD
x_data = [1]
y_data = [1]
=======
x = [1, 2, 3]
y = [1, 2, 3]
>>>>>>> 9ed657a915822d28e59af38b7d7d2956b2aa01b5

#----- a neuron
w = tf.Variable(tf.random_normal([1]))
hypo = w * x_data

#----- learning
cost = (hypo - y_data) ** 2

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1001):
    sess.run(train)

    if i % 100 == 0:
        print('w:', sess.run(w), 'cost:', sess.run(cost))

#----- testing(prediction)
x = [5, 6, 7]
hypo2 = w * x

print(sess.run(hypo2))

