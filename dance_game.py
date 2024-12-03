import time
import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
from random import choice
import pygame

class DancePoseGame:
    def __init__(self):
        # Model Initialization
        self.model = YOLO('yolov8n.pt')

        # MediaPipe Pose Detection Setup
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Game Configuration
        self.POSITIONS = ["prayer", "hands_left", "hands_right",
                          "hands_hips", "hands_up", "hands_separated"]

        self.MUSIC_CHOICES = {
            "left": ("music/waka.mp3", "Waka Waka"),
            "right": ("music/on_the_floor.mp3", "On The Floor")
        }

        self.CHOICE_DURATION = 1.5
        self.TARGET_POSE_TIME = 3

        # Game State
        self.score_tracker = ScoreTracker()
        self.current_pose_index = 0

        # Load sunglasses image
        self.sunglasses = cv2.imread('img/sunglasses.png', cv2.IMREAD_UNCHANGED)
        if self.sunglasses is None:
            print("Warning: Sunglasses image not found!")

    def choose_music(self, cap):
        """Permet au joueur de choisir la musique avec des mouvements de tête."""
        start_time = None
        choice = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            height, width, _ = frame.shape
            middle_x = width // 2  # Position du milieu de l'écran

            # Dessiner la barre verticale au milieu
            cv2.line(frame, (middle_x, 0), (middle_x, height), (0, 255, 255), 2)

            results = self.model(frame)
            detections = results[0].boxes

            for box in detections:
                if box.cls == 0:  # Person class
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    person_roi = frame[y1:y2, x1:x2]
                    person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                    results_pose = self.pose_detector.process(person_rgb)

                    if results_pose.pose_landmarks:
                        nose = results_pose.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE.value]

                        if nose.x < 0.4:  # À gauche
                            if choice != "left":
                                start_time = time.time()
                                choice = "left"
                            elif time.time() - start_time >= self.CHOICE_DURATION:
                                return self.MUSIC_CHOICES["left"]

                        elif nose.x > 0.6:  # À droite
                            if choice != "right":
                                start_time = time.time()
                                choice = "right"
                            elif time.time() - start_time >= self.CHOICE_DURATION:
                                return self.MUSIC_CHOICES["right"]

                        else:
                            start_time = None
                            choice = None

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

                    # Affichage des noms des musiques
                    cv2.putText(frame, "Left: Waka Waka", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "Right: On The Floor", (middle_x + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

            cv2.imshow("Music Selection", frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        return None

    def play_music(self, music_path):
        """Play the selected music using pygame"""
        pygame.mixer.init()
        pygame.mixer.music.load(music_path)
        pygame.mixer.music.play(-1)  # -1 means loop indefinitely

    def stop_music(self):
        """Stop the music"""
        pygame.mixer.music.stop()

    def overlay_sunglasses(self, frame, landmarks, offset_x=20, offset_y=50):
        """
        Place sunglasses precisely on detected eyes with customizable positioning.

        Args:
            frame (numpy.ndarray): The input video frame
            landmarks (list): MediaPipe pose landmarks
            offset_x (int, optional): Horizontal adjustment for sunglasses position
            offset_y (int, optional): Vertical adjustment for sunglasses position

        Returns:
            numpy.ndarray: Frame with sunglasses overlaid
        """
        try:
            # Retrieve eye landmarks
            left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE.value]
            right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE.value]

            # Calculate eye center and dimensions
            eye_center_x = int((left_eye.x + right_eye.x) / 2 * frame.shape[1])
            eye_center_y = int((left_eye.y + right_eye.y) / 2 * frame.shape[0])

            # Calculate eye width, making sunglasses slightly wider
            eye_width = int(np.sqrt(
                (left_eye.x - right_eye.x) ** 2 +
                (left_eye.y - right_eye.y) ** 2
            ) * frame.shape[1] * 2.2)  # Increased multiplier for better coverage

            # Resize sunglasses to fit eye width while maintaining aspect ratio
            if self.sunglasses is not None:
                sunglasses_resized = cv2.resize(
                    self.sunglasses,
                    (eye_width, int(eye_width * self.sunglasses.shape[0] / self.sunglasses.shape[1])),
                    interpolation=cv2.INTER_AREA
                )

                # Get sunglasses dimensions
                h, w, _ = sunglasses_resized.shape

                # Calculate precise positioning with offsets
                top_left_x = eye_center_x - w // 2 + offset_x
                top_left_y = eye_center_y - h // 3 + offset_y

                # Alpha blending for transparent overlay
                for i in range(h):
                    for j in range(w):
                        # Boundary check
                        if (0 <= top_left_y + i < frame.shape[0] and
                                0 <= top_left_x + j < frame.shape[1]):

                            # Get alpha value (transparency)
                            alpha = sunglasses_resized[i, j, 3] / 255.0

                            if alpha > 0:
                                # Blend pixel with background
                                frame[top_left_y + i, top_left_x + j] = (
                                        alpha * sunglasses_resized[i, j, :3] +
                                        (1 - alpha) * frame[top_left_y + i, top_left_x + j]
                                )

            return frame

        except Exception as e:
            print(f"Error overlaying sunglasses: {e}")
            return frame



    def is_good_position(self, landmarks, position):
        """Validate pose correctness based on position"""
        try:
            if position == "prayer":
                left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
                right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
                distance = np.sqrt((left_wrist.x - right_wrist.x) ** 2 +
                                   (left_wrist.y - right_wrist.y) ** 2)
                return distance <= 0.3

            elif position == "hands_right":
                nose_x = landmarks[self.mp_pose.PoseLandmark.NOSE.value].x
                left_wrist_x = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x
                right_wrist_x = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x
                return left_wrist_x <= nose_x and right_wrist_x <= nose_x

            elif position == "hands_left":
                nose_x = landmarks[self.mp_pose.PoseLandmark.NOSE.value].x
                left_wrist_x = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x
                right_wrist_x = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x
                return left_wrist_x >= nose_x and right_wrist_x >= nose_x

            elif position == "hands_hips":
                left_wrist_y = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y
                right_wrist_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y
                left_hip_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
                right_hip_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y
                return left_wrist_y >= left_hip_y and right_wrist_y >= right_hip_y

            elif position == "hands_up":
                left_wrist_y = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y
                right_wrist_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y
                return left_wrist_y <= 0.3 and right_wrist_y <= 0.3

            elif position == "hands_separated":
                left_wrist_x = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x
                right_wrist_x = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x
                return abs(left_wrist_x - right_wrist_x) >= 0.3

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

        selected_music = self.choose_music(cap)
        if selected_music:
            music_path, music_name = selected_music
            print(f"Selected music: {music_name}")
            cv2.destroyWindow("Music Selection")
            self.play_music(music_path)
        else:
            print("No music selected. Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Person detection
            results = self.model(frame)
            detections = results[0].boxes

            # Load target pose image
            pose_image = cv2.imread(f"img/{self.POSITIONS[self.current_pose_index]}.png", cv2.IMREAD_UNCHANGED)
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

                        # Overlay sunglasses
                        frame = self.overlay_sunglasses(frame, landmarks)

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
                            self.mp_pose.POSE_CONNECTIONS
                        )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

            cv2.imshow("Just Dance", frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.stop_music()
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