import numpy as np
import time
from tensorflow.keras.models import load_model
import cv2
from mediapipe_impl.pose_estimation import PoseEstimationModule as pm
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
detector = pm.PoseDetector()

# 加载已保存的模型
model = load_model('./models/pose_estimation.keras')

# 验证模型是否加载成功
model.summary()

ALL_LABELS = ['Blur'], ['Normal'], ['Wrong']
cap = cv2.VideoCapture(1)

p_time = 0
while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        data = []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            # print(id, lm.x)
            data.extend([lm.x, lm.y, lm.z, lm.visibility])
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        X = np.array(data).reshape((1, 33, 4))
        print(X.shape)

        predictions = model.predict(X)

        # 获取预测类别的索引
        predicted_class = np.argmax(predictions, axis=1)[0]

        print(f"Predicted Class: {predicted_class}")
        print(ALL_LABELS[predicted_class])

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(img, 'FPS: {:.2f}'.format(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('img', img)
    cv2.waitKey(1)
