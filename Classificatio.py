## 딥러닝을 활용한 분류 예측 - 와인 품질 등급 판별

import pandas as pd
import numpy as np
import random
import tensorflow as tf

# 램덤 시드 고정
SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 데이터 전처리
drive_path = 'data/'

train = pd.read_csv(drive_path + 'train.csv')
test =  pd.read_csv(drive_path + 'test.csv')
submission = pd.read_csv(drive_path + 'sample_submission.csv')

train['type'].value_counts()
train['type'] = np.where(train['type'] == 'white', 1, 0).astype(int)
test['type'] = np.where(test['type'] == 'white', 1, 0).astype(int)

train['type'].value_counts()
train['quality'].value_counts()

# 범주형 데이터는 원핫인코딩 합니다.
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(train.loc[:, 'quality'] - 3)

# 피처 선택
X_train = train.loc[:, 'fixed acidity':]
X_test = test.loc[:, 'fixed acidity':]

# 피처 스케일링
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# 모델 설계 - 드랍아웃 활용

# 심층 신경망 모델
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_model(train_data, train_target):
    model = Sequential()
    model.add(Dense(128, activation='tanh', input_dim=train_data.shape[1]))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(train_target.shape[1], activation='softmax'))
    
    model.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics=['acc', 'mae'])
    
    return model

model = build_model(X_train_scaled, y_train)
#model.summary()

# 콜백 함수 - Early Stopping 기법
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train
                                            , test_size=0.5
                                            , shuffle=True
                                            , random_state=SEED)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_tr, y_tr
                    , batch_size=64
                    , epochs=200
                    , validation_data=(X_val, y_val)
                    , callbacks=[early_stopping], verbose=2)
                    
model.evaluate(X_val, y_val)
# 예측값 정리 및 파일 제출

# test 데이터에 대한 예측값 정리
y_pred_proba = model.predict(X_test)
y_pred_proba[:5]

y_pred_label = np.argmax(y_pred_proba, axis=-1)+3
y_pred_label[:5]

# 제출 양식에 맞게 정리
submission['quality'] = y_pred_label.astype(int)
submission.to_csv(drive_path + 'wine_dnn_001.csv', index=False)
