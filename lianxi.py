from ultralytics import YOLO

model = YOLO('yolov10n.pt')
# model('A.jpg',show=True,save=True)   # 检测图片，show表示展示，save表示保存

# model.train(data='data.yaml',epochs=80,batch=8,imgsz=640,device='cpu')  # 开始训练模型
# 这里建议训练集：验证集；测试集的比例为6：2：2

# model.val(data='data.yaml', batch=16)   # 开始验证模型

a = YOLO('runs/detect/train4/weights/best.pt')  # 使用训练好的模型
a.predict('xun1/images/test', show=True, save=True)  # 检测文件夹中的图片