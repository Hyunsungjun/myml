# (4)2b-4S/C
import tensorflow as tf

x_data = [[0., 0], [0, 1], [1, 0], [1, 1]]
y_data = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]

#------- 2 inputs 4 neurons
W = tf.Variable(tf.random_normal([2, 4]))
b = tf.Variable(tf.random_normal([4]))
output = tf.matmul(x_data, W) + b

#----- learning
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_data))

train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    sess.run(train)

    if step % 500 == 0:
        print(step, sess.run(cost))

#----- testing(classification)
predicted = tf.equal(tf.argmax(output, 1), tf.argmax(y_data, 1))
accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))

h = sess.run(output)
print("\nLogits: ", h)

p = sess.run(predicted)
print("Predicted: ", p)

a = sess.run(accuracy)
print("Accuracy(%): ", a * 100)
