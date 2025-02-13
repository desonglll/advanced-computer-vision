import cv2
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode, max_num_faces=self.maxFaces,
                                                 min_detection_confidence=self.minDetectionCon,
                                                 min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def find_face_mesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(self.imgRGB)
        faces = []
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_FACE_OVAL, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    # print(id, x, y)
                    face.append([x, y])
                faces.append(face)
        return img, faces

    # cap = cv2.VideoCapture("./datasets/face_videos/face_1.mp4")


def main():
    cap = cv2.VideoCapture(2)
    pTime = 0
    cTime = 0
    detector = FaceMeshDetector(maxFaces=1)

    while True:
        success, img = cap.read()
        img, faces = detector.find_face_mesh(img, False)
        if len(faces) != 0:
            print(len(faces[0]))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(0)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 255), 3)
        cv2.imshow('img', img)
        cv2.waitKey(1)
    pass


if __name__ == '__main__':
    main()
    pass
