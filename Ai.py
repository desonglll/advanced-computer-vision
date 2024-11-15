import json
import os

import cv2
from mediapipe_impl.pose_estimation import PoseEstimationModule as pm

# cap = cv2.VideoCapture(0)
detector = pm.PoseDetector()

images_dir = "./datasets/img"
images = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

for image_path in images:
    print(image_path)
    # success, img = cap.read()
    # img = cv2.resize(img, (1280, 720))
    img = cv2.imread(image_path)
    img = detector.find_pose(img, draw=False)
    lm_list = detector.find_position(img, False)
    cv2.imshow('img', img)
    cv2.waitKey(1)  # 等待1毫秒刷新窗口

    # label = input(f"Please enter the label (0/1): ")
    data = {
        "image": image_path,
        "features": lm_list,
        "label": 1
    }

    # 读取现有JSON文件（如果存在），将新数据追加到列表中
    json_file_path = "data.json"
    try:
        with open(json_file_path, "r") as file:
            data_list = json.load(file)
    except FileNotFoundError:
        data_list = []

    # 追加新数据
    data_list.append(data)

    # 将更新后的列表写回JSON文件
    with open(json_file_path, "w") as file:
        json.dump(data_list, file, indent=4)

    # 关闭窗口并退出循环
    cv2.destroyAllWindows()
