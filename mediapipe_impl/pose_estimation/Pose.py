import cv2
import mediapipe as mp
import math
from typing import Dict


class LandMark:
    def __init__(self, keypoint_id, x, y, z, visibility):
        self.keypoint_id = keypoint_id
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

    def __str__(self):
        return f"keypoint_id: {self.keypoint_id}, x: {self.x}, y: {self.y}, z: {self.z}"

    def __repr__(self):
        return f"keypoint_id: {self.keypoint_id}, x: {self.x}, y: {self.y}, z: {self.z}"

    def convert_point_to_x_y_pixel(self, width, height, channel):
        cx = int(self.x * width)
        cy = int(self.y * height)
        self.x = cx
        self.y = cy


def calculate_angle_with_horizontal(x1, y1, x2, y2):
    # 计算法向量
    nx = y1 - y2
    ny = x2 - x1

    # 计算夹角（使用 atan2）
    angle_radians = math.atan2(ny, nx)  # 法向量与水平线的夹角
    angle_degrees = math.degrees(angle_radians) - 180

    # 确保角度在 0 到 360 度范围
    angle_degrees = (angle_degrees + 360) % 360

    if angle_degrees > 180:
        angle_degrees -= 360

    print(f"法向量与水平线的夹角（弧度）：{angle_radians}")
    print(f"法向量与水平线的夹角（度）：{angle_degrees}")
    return angle_degrees


class PoseDetector:
    def __init__(self, mode=False, smooth=True, detection_con=0.5, track_con=0.5):
        self.results = None
        self.lm_dict: Dict[int, LandMark] = {}
        self.mode = mode
        self.smooth = smooth
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=self.mode, smooth_landmarks=smooth,
                                      min_detection_confidence=detection_con, min_tracking_confidence=track_con)

    def find_pose(self, img, draw=True):
        """
        用于寻找图片内所有的关键点，并将其连线在图中标注出来，可单独使用

        :return: 返回标注所有关键点连线后的img
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, convert_to_x_y_pixel=False, draw=False):
        """
        用于返回图片内所有的关键点和landmarks列表，并将其在图中标注出来，可单独使用

        :return: [{
                    "keypoint_id": id,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility,
                }]
        """
        self.lm_dict = {}
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            for keypoint_id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                raw_landmark = LandMark(keypoint_id, lm.x, lm.y, lm.z, lm.visibility)
                landmark = raw_landmark
                if convert_to_x_y_pixel:
                    landmark.convert_point_to_x_y_pixel(w, h, c)
                self.lm_dict[keypoint_id] = LandMark(keypoint_id, landmark.x, landmark.y, landmark.z,
                                                     landmark.visibility)
                if draw and convert_to_x_y_pixel:
                    cv2.circle(img, (landmark.x, landmark.y), 10, (255, 0, 0), cv2.FILLED)
        return img, self.lm_dict

    def find_angle(self, img, p1, p2, p3, draw=True):
        h, w, c = img.shape
        img, self.lm_dict = self.find_position(img)

        for keypoint_id, lm in self.lm_dict.items():
            lm.convert_point_to_x_y_pixel(w, h, c)
        angle = None
        if p1 in self.lm_dict and p2 in self.lm_dict and p3 in self.lm_dict:
            x1, y1 = self.lm_dict[p1].x, self.lm_dict[p1].y
            x2, y2 = self.lm_dict[p2].x, self.lm_dict[p2].y
            x3, y3 = self.lm_dict[p3].x, self.lm_dict[p3].y
            # Calculate the angle
            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
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

    def find_angle_with_horizontal(self, img, p1, p2, draw=True):
        h, w, c = img.shape
        img, self.lm_dict = self.find_position(img)

        for keypoint_id, lm in self.lm_dict.items():
            lm.convert_point_to_x_y_pixel(w, h, c)
        angle_degrees = None
        if p1 in self.lm_dict and p2 in self.lm_dict:
            x1, y1 = self.lm_dict[p1].x, self.lm_dict[p1].y
            x2, y2 = self.lm_dict[p2].x, self.lm_dict[p2].y

            # 计算方向向量
            dx = x2 - x1
            dy = y2 - y1

            # 计算法向量
            nx = -dy  # 法向量 x 分量
            ny = dx  # 法向量 y 分量

            # 计算法向量与水平线的夹角
            angle_degrees = calculate_angle_with_horizontal(x1, y1, x2, y2)

            if draw:
                # 绘制原直线
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
                cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)

                # 绘制法向量
                fx = x1 + int(nx * 20)  # 调整法向量长度
                fy = y1 + int(ny * 20)
                cv2.arrowedLine(img, (x1, y1), (fx, fy), (0, 255, 0), 3, tipLength=0.3)

                # 标注角度
                cv2.putText(img, f"{int(angle_degrees)} degree", (x1 + 20, y1), cv2.FONT_HERSHEY_PLAIN, 2,
                            (255, 0, 255), 2)

        return angle_degrees

    def find_angle_with_horizontal_mean(self, img, p1, p2, p3, p4, draw=False):
        h, w, c = img.shape
        img, self.lm_dict = self.find_position(img)

        for keypoint_id, lm in self.lm_dict.items():
            lm.convert_point_to_x_y_pixel(w, h, c)
        angle_degrees = None
        cv2.putText(img, f"p1 yes" if p1 in self.lm_dict else "p1 no", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2)
        cv2.putText(img, f"p2 yes" if p2 in self.lm_dict else "p2 no", (20, 90), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2)
        cv2.putText(img, f"p3 yes" if p3 in self.lm_dict else "p3 no", (20, 130), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2)
        cv2.putText(img, f"p4 yes" if p4 in self.lm_dict else "p4 no", (20, 170), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2)
        angle_degrees = None
        if p1 in self.lm_dict and p2 in self.lm_dict and p3 in self.lm_dict and p4 in self.lm_dict:
            x1, y1 = self.lm_dict[p1].x, self.lm_dict[p1].y
            x2, y2 = self.lm_dict[p2].x, self.lm_dict[p2].y
            x3, y3 = self.lm_dict[p3].x, self.lm_dict[p3].y
            x4, y4 = self.lm_dict[p4].x, self.lm_dict[p4].y
            x1_mean = (x1 + x2) / 2
            y1_mean = (y1 + y2) / 2
            x2_mean = (x3 + x4) / 2
            y2_mean = (y3 + y4) / 2
            # 计算方向向量
            dx = x2_mean - x1_mean
            dy = y2_mean - y1_mean

            # 计算法向量
            nx = -dy  # 法向量 x 分量
            ny = dx  # 法向量 y 分量

            # 计算法向量与水平线的夹角
            angle_degrees = calculate_angle_with_horizontal(x1_mean, y1_mean, x2_mean, y2_mean)
            if draw:
                # 绘制原直线
                print(f"x1_mean: {x1_mean}, y1_mean: {y1_mean}")
                x1_mean = int(x1_mean)
                y1_mean = int(y1_mean)
                x2_mean = int(x2_mean)
                y2_mean = int(y2_mean)
                cv2.line(img, (x1_mean, y1_mean), (x2_mean, y2_mean), (255, 255, 255), 3)
                cv2.circle(img, (x1_mean, y1_mean), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x1_mean, y1_mean), 15, (0, 0, 255), 2)
                cv2.circle(img, (x2_mean, y2_mean), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2_mean, y2_mean), 15, (0, 0, 255), 2)

                # 绘制法向量
                fx = x1_mean + int(nx * 20)  # 调整法向量长度
                fy = y1_mean + int(ny * 20)
                cv2.arrowedLine(img, (x1_mean, y1_mean), (fx, fy), (0, 255, 0), 3, tipLength=0.3)

                # 标注角度
                cv2.putText(img, f"{int(angle_degrees)} degree", (x1_mean + 20, y1), cv2.FONT_HERSHEY_PLAIN, 2,
                            (255, 0, 255), 2)

        return angle_degrees

    # TODO:
    def find_angle_with_horizontal_mean_all(self, img, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, draw=False):
        h, w, c = img.shape
        img, self.lm_dict = self.find_position(img)

        for keypoint_id, lm in self.lm_dict.items():
            lm.convert_point_to_x_y_pixel(w, h, c)
        angle_degrees = None
        cv2.putText(img, f"p1 yes" if p1 in self.lm_dict else "p1 no", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2)
        cv2.putText(img, f"p2 yes" if p2 in self.lm_dict else "p2 no", (20, 90), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2)
        cv2.putText(img, f"p3 yes" if p3 in self.lm_dict else "p3 no", (20, 130), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2)
        cv2.putText(img, f"p4 yes" if p4 in self.lm_dict else "p4 no", (20, 170), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2)
        cv2.putText(img, f"p5 yes" if p5 in self.lm_dict else "p5 no", (20, 210), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2)
        cv2.putText(img, f"p6 yes" if p6 in self.lm_dict else "p6 no", (20, 250), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2)
        cv2.putText(img, f"p7 yes" if p7 in self.lm_dict else "p7 no", (20, 290), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2)
        cv2.putText(img, f"p8 yes" if p8 in self.lm_dict else "p8 no", (20, 330), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2)
        cv2.putText(img, f"p9 yes" if p9 in self.lm_dict else "p9 no", (20, 370), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2)
        cv2.putText(img, f"p10 yes" if p10 in self.lm_dict else "p10 no", (20, 410), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2)
        if p1 in self.lm_dict and p2 in self.lm_dict and p3 in self.lm_dict and p4 in self.lm_dict and p5 in self.lm_dict and p6 in self.lm_dict and p7 in self.lm_dict and p8 in self.lm_dict and p9 in self.lm_dict and p10 in self.lm_dict:
            x1, y1 = self.lm_dict[p1].x, self.lm_dict[p1].y
            x2, y2 = self.lm_dict[p2].x, self.lm_dict[p2].y
            x3, y3 = self.lm_dict[p3].x, self.lm_dict[p3].y
            x4, y4 = self.lm_dict[p4].x, self.lm_dict[p4].y
            x5, y5 = self.lm_dict[p5].x, self.lm_dict[p5].y
            x6, y6 = self.lm_dict[p6].x, self.lm_dict[p6].y
            x7, y7 = self.lm_dict[p7].x, self.lm_dict[p7].y
            x8, y8 = self.lm_dict[p8].x, self.lm_dict[p8].y
            x9, y9 = self.lm_dict[p9].x, self.lm_dict[p9].y
            x10, y10 = self.lm_dict[p10].x, self.lm_dict[p10].y
            x14_mean = (x1 + x4) / 2
            y14_mean = (y1 + y4) / 2
            x25_mean = (x2 + x5) / 2
            y25_mean = (y2 + y5) / 2
            x36_mean = (x3 + x6) / 2
            y36_mean = (y3 + y6) / 2
            x78_mean = (x7 + x8) / 2
            y78_mean = (y7 + y8) / 2
            x910_mean = (x9 + x10) / 2
            y910_mean = (y9 + y10) / 2

            x1_mean = (x14_mean + x25_mean + x36_mean + x78_mean) / 4
            y1_mean = (y14_mean + y25_mean + y36_mean + y78_mean) / 4

            x2_mean = x910_mean
            y2_mean = y910_mean
            # 计算方向向量
            dx = x2_mean - x1_mean
            dy = y2_mean - y1_mean

            # 计算法向量
            nx = -dy  # 法向量 x 分量
            ny = dx  # 法向量 y 分量

            # 计算法向量与水平线的夹角
            angle_degrees = calculate_angle_with_horizontal(x1_mean, y1_mean, x2_mean, y2_mean) - 10
            if draw:
                # 绘制原直线
                print(f"x1_mean: {x1_mean}, y1_mean: {y1_mean}")
                x1_mean = int(x1_mean)
                y1_mean = int(y1_mean)
                x2_mean = int(x2_mean)
                y2_mean = int(y2_mean)
                cv2.line(img, (x1_mean, y1_mean), (x2_mean, y2_mean), (255, 255, 255), 3)
                cv2.circle(img, (x1_mean, y1_mean), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x1_mean, y1_mean), 15, (0, 0, 255), 2)
                cv2.circle(img, (x2_mean, y2_mean), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2_mean, y2_mean), 15, (0, 0, 255), 2)

                # 绘制法向量
                fx = x1_mean + int(nx * 20)  # 调整法向量长度
                fy = y1_mean + int(ny * 20)
                cv2.arrowedLine(img, (x1_mean, y1_mean), (fx, fy), (0, 255, 0), 3, tipLength=0.3)

                # 标注角度
                cv2.putText(img, f"{int(angle_degrees)} degree", (x1_mean + 20, y1), cv2.FONT_HERSHEY_PLAIN, 2,
                            (255, 0, 255), 2)

        return angle_degrees


def main():
    # cap = cv2.VideoCapture("../../datasets/body_videos/body_1.jpg")
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.find_pose(img)
        cv2.imshow('img', img)

        # img, lmks = detector.find_position(img, convert_to_x_y_pixel=True, draw=True)
        # cv2.imshow('img', img)
        # print(lmks)

        # angle = detector.find_angle(img, 16, 14, 12)
        # cv2.imshow('img', img)
        # print(angle)

        # angle = detector.find_angle_with_horizontal(img, 5, 10, draw=True)
        # cv2.imshow('img', img)
        #
        # detector.find_angle_with_horizontal_mean(img, 2, 5, 9, 10, draw=True)
        # cv2.imshow('img', img)

        # detector.find_angle_with_horizontal_mean_all(img, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, draw=True)
        # cv2.imshow('img', img)
        # TODO:
        # 调用方法，处理图像并定期保存角度
        # img = detector.process_and_save_angle(
        #     img=img,
        #     p1=5, p2=12, p3=24,
        #     save_interval=1,  # 每1秒保存一次
        #     filepath="angles.csv",
        #     draw=True
        # )
        #
        # 显示图像
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    pass
