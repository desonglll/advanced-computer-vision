import cv2
import mediapipe as mp
import math


class LandMark:
    def __init__(self, keypoint_id, x, y, z, visibility):
        self.keypoint_id = keypoint_id
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

    def __str__(self):
        return f"keypoint_id: {self.keypoint_id}, x: {self.x}, y: {self.y}, z: {self.z}"

    def convert_point_to_x_y_pixel(self, width, height, channel):
        cx = int(self.x * width)
        cy = int(self.y * height)
        self.x = cx
        self.y = cy


class PoseDetector:
    def __init__(self, mode=False, up_body=False, smooth=True, detection_con=0.5, track_con=0.5):
        self.results = None
        self.lm_list = [LandMark]
        self.lm_map = None
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
        self.lm_list = []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            for keypoint_id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                raw_landmark = LandMark(keypoint_id, lm.x, lm.y, lm.z, lm.visibility)
                landmark = raw_landmark
                if convert_to_x_y_pixel:
                    landmark.convert_point_to_x_y_pixel(w, h, c)
                self.lm_list.append(LandMark(keypoint_id, landmark.x, landmark.y, landmark.z, landmark.visibility))
                # self.lm_list.append({
                #     "keypoint_id": landmark.keypoint_id,
                #     "x": landmark.x,
                #     "y": landmark.y,
                #     "z": landmark.z,
                #     "visibility": landmark.visibility,
                # })
                if draw and convert_to_x_y_pixel:
                    cv2.circle(img, (landmark.x, landmark.y), 10, (255, 0, 0), cv2.FILLED)
        return img, self.lm_list

    def calculate_angle_with_horizontal(self, x1, y1, x2, y2):
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

    def find_angle_with_horizontal(self, img, p1, p2, draw=True):
        h, w, c = img.shape
        img, self.lm_list = self.find_position(img)

        for lm in self.lm_list:
            lm.convert_point_to_x_y_pixel(w, h, c)
        x1, y1 = self.lm_list[p1].x, self.lm_list[p1].y
        x2, y2 = self.lm_list[p2].x, self.lm_list[p2].y

        # 计算方向向量
        dx = x2 - x1
        dy = y2 - y1

        # 计算法向量
        nx = -dy  # 法向量 x 分量
        ny = dx  # 法向量 y 分量

        # 计算法向量与水平线的夹角
        angle_radians = math.atan2(ny, nx)
        angle_degrees = math.degrees(angle_radians)

        if draw:
            # 绘制原直线
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)

            # 绘制法向量
            fx = x1 + int(nx * 50)  # 调整法向量长度
            fy = y1 + int(ny * 50)
            cv2.arrowedLine(img, (x1, y1), (fx, fy), (0, 255, 0), 3, tipLength=0.3)

            # 标注角度
            cv2.putText(img, f"{int(angle_degrees)}°", (fx + 10, fy), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return angle_degrees

    def find_angle(self, img, p1, p2, p3, draw=True):
        h, w, c = img.shape
        img, self.lm_list = self.find_position(img)

        for lm in self.lm_list:
            lm.convert_point_to_x_y_pixel(w, h, c)
        # print(id, lm)
        if self.lm_list[p1] and self.lm_list[p2] and self.lm_list[p3]:
            x1, y1 = self.lm_list[p1].x, self.lm_list[p1].y
            x2, y2 = self.lm_list[p2].x, self.lm_list[p2].y
            x3, y3 = self.lm_list[p3].x, self.lm_list[p3].y

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
    #
    # def process_and_save_angle(self, img, p1, p2, p3, save_interval, filepath="angles.csv", draw=True):
    #     """
    #     处理图像，计算角度，并定期保存角度和时间戳到CSV文件中。
    #     :param img: 输入图像
    #     :param p1: 第一个关键点ID
    #     :param p2: 第二个关键点ID
    #     :param p3: 第三个关键点ID
    #     :param save_interval: 保存间隔时间（秒）
    #     :param filepath: CSV文件路径
    #     :param draw: 是否在图像上绘制关键点和角度
    #     :return: 处理后的图像
    #
    #     csv格式：time, angle
    #     """
    #     # 初始化静态变量，用于存储上次保存时间
    #     if not hasattr(self, "_last_save_time"):
    #         self._last_save_time = 0
    #
    #     lm_list = self.find_position(img, draw=False)
    #
    #     if len(lm_list) > 0:
    #         # 计算三个关键点的角度
    #         angle = self.find_angle(img, p1, p2, p3, draw=draw)
    #
    #         # 获取当前时间
    #         current_time = time.time()
    #
    #         # 检查是否需要保存角度
    #         if current_time - self._last_save_time >= save_interval:
    #             # 保存到CSV文件
    #             with open(filepath, mode='a', newline='') as file:
    #                 writer = csv.writer(file)
    #                 writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)), angle])
    #             print(
    #                 f"Angle {angle:.2f} saved at time {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}.")
    #
    #             # 更新保存时间
    #             self._last_save_time = current_time
    #
    #     return img


def main():
    # cap = cv2.VideoCapture("../../datasets/body_videos/body_1.jpg")
    cap = cv2.VideoCapture(2)
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        # img = detector.find_pose(img)
        # cv2.imshow('img', img)

        img, lmks = detector.find_position(img, convert_to_x_y_pixel=True, draw=True)
        cv2.imshow('img', img)
        print(lmks)

        # angle = detector.find_angle(img, 16, 14, 12)
        # cv2.imshow('img', img)
        # print(angle)
        #
        # angle = detector.find_angle_with_horizontal(img, 5, 10)
        # cv2.imshow('img', img)
        # print(angle)

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
