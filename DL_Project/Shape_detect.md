# Deep_Project1day
## Theme :  Handwritten Triangle / Circle / Star / Rectangle Shape Classification (Training)
---
### Prerequisites
- 파이썬 프로그래밍 언어에 특화된 컴퓨터 프로그래밍에 사용되는 통합 개발 환경
- ex) VS code , Google Colab , Jupyter notebook , PyCharm , anaconda
- "I couldn't upload the data file because it's too large. For fun, let's create the data ourselves!"

### pip install basic modules 
(Take a look at what modules are needed based on the model you're using.)

```sh
$ pip install tensorflow
$ pip install torch torchvision
$ pip install numpy
$ pip install matplotlib
$ pip install seaborn
$ pip install plotly
$ pip install scikit-learn
$ pip install pandas
$ pip install opencv-python

```

### example shape picture
![image](https://github.com/AUTO-KKYU/Deep_Project1day/assets/118419026/6c813747-353c-46e3-8e2a-a59d9152442a)
![image](https://github.com/AUTO-KKYU/Deep_Project1day/assets/118419026/92f1bf9c-1d86-462f-a510-e8397cd2762e)
![image](https://github.com/AUTO-KKYU/Deep_Project1day/assets/118419026/1f9f2e59-2185-4d2d-b088-d65fa2131e15)
![image](https://github.com/AUTO-KKYU/Deep_Project1day/assets/118419026/4b36a638-cf73-4e6d-ae97-079814db9492)


### The model used and evaluation results
#### "The results of training the model using my data."

| Model | Train_Accuracy | Test_Accuracy |
| --- | --- | --- |
| `InceptionV3` | 0.9958 | 0.9250 | 
| `VGG16` | 0.9109 | 0.9375 | 
| `ResNet50` | 0.9986 | 0.8625 | 
| `CNN` | 0.9541 | 0.8750 | 
| `Xception` | 1.0000 | 1.0000 | 

### EfficientNet 모델 
- 딥러닝 기반의 이미지 분류 작업에서 큰 인기를 얻고 있는 효율적인 모델
- CNN 기반 모델
- Compound Scaling은 네트워크의 깊이, 너비, 해상도의 크기를 동시에 확장시키는 방법으로, 세 가지의 요소를 조절하여 최적의 모델 구조를 찾아냄
  
![Screenshot from 2024-03-02 16-40-37](https://github.com/AUTO-KKYU/Deep_Project1day/assets/118419026/ca2d73ce-e1ac-4e86-9a2a-8b7abc390215)

![image](https://github.com/AUTO-KKYU/Deep_Project1day/assets/118419026/f7cfcc8f-10e9-428e-bcc3-7d36683bf7a3)





---
### 느낀점
- 데이터는 많을수록 좋다
- 모델 선택도 중요한 부분도 있지만 데이터 전처리 과정이 매우 중요하다.
- 모델이 과적합이 되었을수도 있기 때문에 모델의 복잡도를 줄여봐야겠다
- data augmentation / fallback / learning rate / pre-trained model / cross_validation
    => 해당 옵션들에 대해 다양하게 시도해보는 것이 좋다
  

#### Notion Link
[Notion 페이지](https://www.notion.so/eb580b8e526d4246a3b80c2a256fd6cb?pvs=4)

#### 모델 구성 참고자료
https://github.com/keras-team/keras-docs-ko/blob/master/sources/getting-started/sequential-model-guide.md
