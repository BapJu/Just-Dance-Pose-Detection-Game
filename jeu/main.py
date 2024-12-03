import time
import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
from random import choice


class DancePoseGame:
    def __init__(self):
        # Model Initialization
        self.model = YOLO('yolov8n.pt')

        # MediaPipe Pose Detection Setup
        mp_pose = mp.solutions.pose
        self.pose_detector = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Game Configuration
        self.POSITIONS = ["priere", "mains_a_gauche", "mains_a_droite",
                          "mains_fesses", "main_en_air", "mains_ecartees"]
        self.TARGET_POSE_TIME = 1.5

        # Game State
        self.score_tracker = ScoreTracker()
        self.current_pose_index = 0

    def is_good_position(self, landmarks, position):
        """Validate pose correctness based on position"""
        try:
            if position == "priere":
                left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
                right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
                distance = np.sqrt((left_wrist.x - right_wrist.x) ** 2 +
                                   (left_wrist.y - right_wrist.y) ** 2)
                return distance <= 0.3

            elif position == "mains_a_droite":
                nose_x = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value].x
                left_wrist_x = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x
                right_wrist_x = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x
                return left_wrist_x <= nose_x and right_wrist_x <= nose_x

            # Add other position checks similarly

            return False
        except Exception as e:
            print(f"Error in pose validation: {e}")
            return False

    def overlay_pose_image(self, frame, pose_image):
        """Overlay pose image with transparency"""
        if pose_image.shape[2] == 4:
            img_rgb = pose_image[:, :, :3]
            img_alpha = pose_image[:, :, 3] / 255.0

            height, width, _ = frame.shape
            pose_height, pose_width, _ = img_rgb.shape
            x_offset = width - pose_width - 10
            y_offset = height - pose_height - 10

            for c in range(3):
                frame[y_offset:y_offset + pose_height, x_offset:x_offset + pose_width, c] = \
                    (1. - img_alpha) * frame[y_offset:y_offset + pose_height, x_offset:x_offset + pose_width, c] + \
                    img_alpha * img_rgb[:, :, c]
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Person detection
            results = self.model(frame)
            detections = results[0].boxes

            # Load target pose image
            pose_image = cv2.imread(f"poses/{self.POSITIONS[self.current_pose_index]}.png", cv2.IMREAD_UNCHANGED)
            frame = self.overlay_pose_image(frame, pose_image)

            for box in detections:
                if box.cls == 0:  # Person class
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)

                    person_roi = frame[y1:y2, x1:x2]
                    person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)

                    # Pose detection
                    results_pose = self.pose_detector.process(person_rgb)

                    # Display current pose and score
                    cv2.putText(frame, self.POSITIONS[self.current_pose_index],
                                (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Score: {self.score_tracker.get_score()}",
                                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    if results_pose.pose_landmarks:
                        landmarks = results_pose.pose_landmarks.landmark
                        is_correct = self.is_good_position(landmarks, self.POSITIONS[self.current_pose_index])

                        # Update score and change pose
                        if self.score_tracker.update_score(self.POSITIONS[self.current_pose_index], is_correct):
                            self.current_pose_index = self.POSITIONS.index(choice(self.POSITIONS))

                        status_color = (0, 255, 0) if is_correct else (0, 0, 255)
                        status_text = "Pose correcte!" if is_correct else "Essaye encore!"
                        cv2.putText(frame, status_text, (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

                        # Draw pose landmarks
                        mp.solutions.drawing_utils.draw_landmarks(
                            person_roi,
                            results_pose.pose_landmarks,
                            mp.solutions.pose.POSE_CONNECTIONS
                        )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

            cv2.imshow("Just Dance", frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


class ScoreTracker:
    def __init__(self):
        self.score = 0
        self.last_valid_pose = None
        self.pose_start_time = None

    def update_score(self, position, is_correct):
        current_time = time.time()

        if is_correct and (self.last_valid_pose != position):
            if self.pose_start_time is None:
                self.pose_start_time = current_time

            if current_time - self.pose_start_time >= 1.5:
                self.score += 100
                self.last_valid_pose = position
                self.pose_start_time = None
                return True
        else:
            self.pose_start_time = None

        return False

    def get_score(self):
        return int(self.score)


def main():
    game = DancePoseGame()
    game.run()


if __name__ == "__main__":
    main()