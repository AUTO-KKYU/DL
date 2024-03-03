## 1번째 방법
### 기존에 가지고 있는 nvidia setting 초기화
sudo apt-get remove --purge nvidia*
sudo apt-get autoremove
sudo apt-get autoclean

sudo dpkg -l | grep nvidia
sudo purge nvidia-settings~~

### ubuntu device list를 최신화로 갱신
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:graphics-drivers/ppa -y

sudo apt-get update

### 설치 가능한 드라이버를 확인하고 원하는 드라이버 설치
sudo ubuntu-drivers devices

sudo apt install nvidia-driver-515 => 선택

### roboot을 해야 설치된 드라이버가 인식함
sudo reboot now 

### 드라이버가 잘 설치되었는지 확인
sudo lshw -c display

### configureation에서 driver-nvidia가 나오면 잘 설치가 됨

### 결과창이 잘 확인이 되면 잘 됨
nvidia-smi

## 2번쨰 방법
1) software update를 클릭
2)additional drivers 누르고 설치하고자 하는 드라이브를 apply
ex) using nvidia driver metapackage from nvidia-driver-515
3) 재부팅

## 3번째 방법
- 추천하는 드라이버 설치 방법
ubuntu-drivers devices

sudo ubuntu-drivers autoinstall

sudo reboot now

### 설치 잘 되었는지 확인
sudo lshw -c display

nvidia-smi

## Cuda 설치
1) google에서 cuda toolkit archive에 들어가기
2) cuda toolkit 11.7 
linux / x86_64 / ubuntu / 22.04 / runfile(local)
3) 명령어들을 한줄씩 입력
continue / driver space바를 눌러 해제 / install가서 enter
nvidia-smi 누르면 cuda version이 뜨는걸 확인할 수 있음

## cudnn 설치
download cudnn v8.9.2 for cuda 11.x 클릭
local installer for linux_x86_64(Tar) 다운

### 다운로드 받은 위치로 이동
cd Downloads/

### 압축 해제
tar xvf cudnn-linux-x86_64-8.9.0.131_cuda11-archive.tar.xz

### 필요한 파일을 각각의 경로에 복사
sudo cp cudnn-linux-x86_64-8.9.0.131_cuda11-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-linux-x86_64-8.9.0.131_cuda11-archive/lib/libcudnn* /usr/local/cuda/lib64

sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

sudo nano ~/.bashrc

export되어있는 부분 전부 추가 후 저장

export CUDA_HOME=/usr/local/cuda-11.7
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PAYH:/usr/local/cuda-11.7/lib64/

source ~/.bashrc
nvidia-smi 
nvcc -V

/usr/local/cuda-11.7/extras/demo_suite/deviceQuery
