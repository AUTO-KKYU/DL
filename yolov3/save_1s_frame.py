import cv2
filepath = '/home/kkyu/amr_ws/yolo/data/soccer_run.mp4 (720p).mp4'
video = cv2.VideoCapture(filepath) #수정

if not video.isOpened():
    print("Video is unavailable :", filepath)
    exit(0)

length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)
count = 0

while(video.isOpened()):
    ret, image = video.read()

    if(int(video.get(1)) % int(fps) == 0):
        cv2.imwrite(filepath[:-4] + "/frame%d.jpg" % count, image)
        print('Saved frame number :', str(video.get(1)))
        count += 1
    if int(video.get(1)) == length:
        video.release()
        break