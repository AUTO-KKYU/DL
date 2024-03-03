## 딥러닝을 활용한 회귀 분석 - 보스턴 주택 가격 예측

import pandas as pd
import numpy as np
import random
import tensorflow as tf

# 랜덤 시드 고정
SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 데이터 전처리
from sklearn import datasets

housing = datasets.load_boston()
X_data = housing.data
y_data = housing.target

# 피처 스케일링
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_data_scaled = scaler.fit_transform(X_data)

X_data_scaled[0]

# 학습 - 테스트 데이터셋 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=True, random_state = SEED)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# MLP 모델 아키텍처 정의

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def build_model(num_input=1):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=num_input))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

model = build_model(num_input=13)

# 미니 배치 학습
model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=2)

# 모델 평가
model.evaluate(X_test, y_test)

# 교차 검증
model = build_model(num_input=13)
history = model.fit(X_train, y_train, batch_size=32, epochs=200, validation_split=0.25, verbose=2)

# 
import matplotlib.pyplot as plt

def plot_loss_curve(total_epoch=10, start=1):
    plt.figure(figsize=(15, 5))
    plt.plot(range(start, total_epoch +1), history.history['loss'][start-1:total_epoch], label='Train')
    plt.plot(range(start, total_epoch +1), history.history['val_loss'][start-1:total_epoch], label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('mse')
    plt.legend()
    plt.show()
    
plot_loss_curve(total_epoch=200, start=1)

plot_loss_curve(total_epoch=200, start=20)
