from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

x = [1]
y = [1]

model = Sequential()

'''
Dense는 완전연결계층, 첫번째 매개변수는 노드의 갯수입니다. 
그리고 input_shape는 입력값의 모양입니다. 
배열의 차원이라고 하면 이해가 쉬울 것 입니다. 
(1,)라고 되어있는 이유는 1개씩 여러번 해야하기에 정하지 않았다는 의미
'''
model.add(Dense(1, input_dim=1))

# MSE because we want linear regression.
'''
compile은 모델을 어떻게 학습할지 정하는 곳입니다. 
loss나 optimizer는 상황에 따라 사용하고 싶은 것을 사용
'''
model.compile(optimizer=SGD(lr=0.1), loss='mse')
'''
fit은 모델을 학습시키는 것. X는 우리가 알수있는 데이터가 들어갑니다. 
input_shape와 맞춰주어야합니다. 
Y는 우리가 X로 예측하고 싶은 정답
'''
model.fit(x, y, nb_epoch=1000, batch_size=1)
print(model.get_weights())

print(model.predict([2]))
