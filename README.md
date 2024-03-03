# DL (Deep Learning)
## 딥러닝이란?
- 딥러닝은 입력과 출력의 관계를 잘 설명할 수 있는 복잡한 함수식의 가중치를 찾는 과정
- 딥러닝 학습은 손실 함수를 최소화하는 인공신경망의 가중치와 편향을 찾는 과정

## 활성화 함수 (Activation Function)
다양한 뉴런에서 데이터를 연산하고 다음 뉴런로 값을 전달해주며, 이 데이터들을 연산하면서 정리해주는 함수
![Screenshot from 2024-03-03 18-22-31](https://github.com/AUTO-KKYU/DL/assets/118419026/4145abdc-19f7-48e4-a3a0-5e1a22fa0b9d)

< 용도 >
- Softmax : 보통 출력층에 사용
- Sigmoid : 중간 activation function에 사용

## 손실 함수 (Cost Function)
![Screenshot from 2024-03-03 18-24-38](https://github.com/AUTO-KKYU/DL/assets/118419026/0b37d613-a3f6-4bb5-9330-c1623cebb4e9)

< 용도 >
- 회귀 모델 : MSE , MAE, RMES
- 분류 모델 : Binary cross-entropy , Categorical cross-entropy

## 최적화 알고리즘 (Optimizer)
![image](https://github.com/AUTO-KKYU/DL/assets/118419026/38b77ee4-0baa-4be4-9854-de1a11f26f06)



## 신경망으로 해결할 수 있는 문제
- Classification (분류)
1) Classification에서는 보통 데이터를 구성하는 클래스 개수와 동일하게 Output 레이어의 퍼셉트론의 개수를 정합니다.
2) 어떤 문제를 만났는데 그 문제에서 추측하고 싶은 결과가 이름 혹은 문자

### "가지고 있는 데이터에 독립변수와 종속변수가 있고 종속변수가 이름일 때 분류 이용 "

- Regression (회귀)
1) 예측하고 싶은 종속변수가 숫자일 때 보통 회귀라는 머신러닝의 방법을 사용
2) 어떤 문제를 만났는데 그 문제에서 예측하고 싶은 결과가 숫자라면 이렇게 하면 된다

### "가지고 있는 데이터에 독립변수와 종속변수가 있고 종속변수가 숫자일 때 회귀 이용 "

![Screenshot from 2024-03-03 18-18-43](https://github.com/AUTO-KKYU/DL/assets/118419026/2127706e-f71f-4aef-bfe0-10e898b18d7d)
