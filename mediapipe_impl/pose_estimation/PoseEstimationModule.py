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
        _, x1, y1 = self.lm_list[p1]
        # x1, y1 = self.lm_list[p1][1:] # Also works
        _, x2, y2 = self.lm_list[p2]
        _, x3, y3 = self.lm_list[p3]
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


def main():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("../../datasets/body_videos/body_2.mp4")
    p_time = 0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.find_pose(img=img)
        lm_list = detector.find_position(img=img, draw=False)
        # if len(lm_list) != 0:
        #     print(lm_list[14])
        #     cv2.circle(img, (lm_list[14][1], lm_list[14][2]), 10, (255, 0, 0), cv2.FILLED)
        #
        c_time = time.time()

        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, 'FPS: {:.2f}'.format(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow('img', img)
        cv2.waitKey(1)

    pass


if __name__ == '__main__':
    main()
    pass
