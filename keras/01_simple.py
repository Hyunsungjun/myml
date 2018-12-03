from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

x_data = [1,2]
y_data = [1,2]

model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.compile(optimizer=SGD(lr=0.1), loss='mse')
model.fit(x_data, y_data, nb_epoch=1000, batch_size=1)

print(model.predict([3, 4]))
