import csv
from time import localtime

import cv2
import mediapipe as mp
import time
import math


class PoseDetector:
    def __init__(self, mode=False, up_body=False, smooth=True, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.up_body = up_body
        self.smooth = smooth
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=self.mode, smooth_landmarks=smooth,
                                      min_detection_confidence=detection_con, min_tracking_confidence=track_con)

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        # print(results.pose_landmarks)
        self.lm_list = []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append({
                    "keypoint_id": id,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility,
                })
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return self.lm_list

    def find_angle(self, img, p1, p2, p3, draw=True):
        h, w, c = img.shape
        # print(id, lm)
        x1, y1 = int(self.lm_list[p1]["x"] * w), int(self.lm_list[p1]["y"] * h)
        x2, y2 = int(self.lm_list[p2]["x"] * w), int(self.lm_list[p2]["y"] * h)
        x3, y3 = int(self.lm_list[p3]["x"] * w), int(self.lm_list[p3]["y"] * h)

        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        # print(angle)
        if angle < 0:
            angle += 360

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return angle

    def process_and_save_angle(self, img, p1, p2, p3, save_interval, filepath="angles.csv", draw=True):
        """
        处理图像，计算角度，并定期保存角度和时间戳到CSV文件中。
        :param img: 输入图像
        :param p1: 第一个关键点ID
        :param p2: 第二个关键点ID
        :param p3: 第三个关键点ID
        :param save_interval: 保存间隔时间（秒）
        :param filepath: CSV文件路径
        :param draw: 是否在图像上绘制关键点和角度
        :return: 处理后的图像

        csv格式：time, angle
        """
        # 初始化静态变量，用于存储上次保存时间
        if not hasattr(self, "_last_save_time"):
            self._last_save_time = 0

        lm_list = self.find_position(img, draw=False)

        if len(lm_list) > 0:
            # 计算三个关键点的角度
            angle = self.find_angle(img, p1, p2, p3, draw=draw)

            # 获取当前时间
            current_time = time.time()

            # 检查是否需要保存角度
            if current_time - self._last_save_time >= save_interval:
                # 保存到CSV文件
                with open(filepath, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)), angle])
                print(
                    f"Angle {angle:.2f} saved at time {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}.")

                # 更新保存时间
                self._last_save_time = current_time

        return img


def main():
    # cap = cv2.VideoCapture("../../datasets/body_videos/body_2.mp4")
    cap = cv2.VideoCapture(1)
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        # 调用方法，处理图像并定期保存角度
        img = detector.process_and_save_angle(
            img=img,
            p1=5, p2=12, p3=24,  # 左肩、左肘、左手腕
            save_interval=1,  # 每1秒保存一次
            filepath="angles.csv",
            draw=True
        )

        # 显示图像
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    pass
