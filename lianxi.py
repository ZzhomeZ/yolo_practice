from ultralytics import YOLO

model = YOLO('yolov10n.pt')
# model('A.jpg',show=True,save=True)   # 检测图片，show表示展示，save表示保存
