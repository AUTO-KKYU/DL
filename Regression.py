## 간단한 딥러닝 모델(회귀) 만들기

import tensorflow as tf
import pandas as pd
import numpy as np

x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-2, 32, -10, 5, 1, 23, -1, -4, -24, -13]

X_train = np.array(x).reshape(-1, 1)
y_train = np.array(y)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(units=1, activation='linear', input_dim=1))

# model.summary()

# 모델 컴파일
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 모델 학습 및 예측
model.fit(X_train, y_train, epochs=3000, verbose=0)

# 모델 가중치 확인
model.weights

# 모델 예측
model.predict([[11], [12], [13]])
