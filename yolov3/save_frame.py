import cv2
filepath = '/home/kkyu/amr_ws/yolo/data/soccer_run.mp4 (720p).mp4'
video = cv2.VideoCapture(filepath) #수정

if not video.isOpened(): #video에 접근 가능한지 확인 -> True/False로 반환
    print("Video is unavailable :", filepath)
    exit(0)

##이미지 저장 파일 생성 코드 추가
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

while(video.isOpened()):
    ret, image = video.read() # while문을 돌면서 frame 단위로 image로 읽어옴
    cv2.imwrite(filepath[:-4] + "/frame%d.jpg" % video.get(1), image) # 읽어온 image를 저장
    print('Saved frame number :', str(int(video.get(1))))

    if int(video.get(1)) == length: # 현재 프레임이 마지막 프레임에 도달하였을 때
        video.release() # video를 로드하느라 사용한 메모리 할당을 해제하고
        break # while문을 빠져나옴
