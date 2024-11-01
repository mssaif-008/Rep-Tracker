import cv2 as cv
import mediapipe as mp
import time
import os

class PoseDetector:
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=1,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)
        self.results = None

    def findPose(self, frame, draw=True, scale=0.5):
        imgRgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRgb)

        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        frame = self.rescale_frame(frame, scale)
        return frame

    @staticmethod
    def rescale_frame(frame, scale=0.5):
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

    def findPosition(self, frame, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(frame, (cx, cy), 4, (255, 0, 0), cv.FILLED)
        return lmList


def detect_exercise_type(file_name):
    if "pushup" in file_name.lower():
        return "pushup"
    elif "shoulderpress" in file_name.lower():
        return "shoulderpress"
    elif "bicep" in file_name.lower():
        return "bicep"
    return None


def count_reps(exercise, lmList, prev_state, rep_count):
    if exercise == "pushup":

        if lmList[13][2] > lmList[11][2] and prev_state == "up":
            rep_count += 1
            prev_state = "down"
        elif lmList[13][2] < lmList[11][2]:
            prev_state = "up"


    elif exercise == "shoulderpress":

        if lmList[13][2] < lmList[11][2] and prev_state == "down":

            prev_state = "up"
            rep_count += 1
        elif lmList[13][2] > lmList[11][2]:
            prev_state = "down"


    elif exercise == "bicep":

        if lmList[15][2] < lmList[13][2] and prev_state == "down":

            prev_state = "up"
            rep_count += 1
        elif lmList[15][2] > lmList[13][2]:
            prev_state = "down"


    return rep_count, prev_state


def main():
    path="PoseVideos/bicep.mp4"
    cap = cv.VideoCapture(path)
    file_name = os.path.basename(path)
    exercise_type = detect_exercise_type(file_name)
    if exercise_type is None:
        print("No recognized exercise type in filename.")
        return

    pTime = 0
    rep_count = 0
    prev_state = "down"
    detector = PoseDetector()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = detector.findPose(frame, scale=0.5)
        lmList = detector.findPosition(frame, draw=False)

        if lmList:
            rep_count, prev_state = count_reps(exercise_type, lmList, prev_state, rep_count)
            cv.putText(frame, f'Reps: {rep_count}', (50, 100), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            cv.putText(frame, f'Exercise: {exercise_type.capitalize()}', (50, 150), cv.FONT_HERSHEY_PLAIN, 2,
                       (255, 0, 0), 3)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime - pTime != 0 else 0
        pTime = cTime
        cv.putText(frame, f'FPS: {int(fps)}', (70, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

        cv.imshow('Image', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
