import cv2

filepath = '/home/kkyu/amr_ws/yolo/data/soccer_run.mp4 (720p).mp4' #수정
video = cv2.VideoCapture(filepath)

if not video.isOpened():
    print("Video is unavailable :", filepath)
    exit(0)

length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

print("length :", length)
print("width :", width)
print("height :", height)
print("fps :", fps)