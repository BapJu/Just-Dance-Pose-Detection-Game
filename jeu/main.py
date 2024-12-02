import time
import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
from random import randint

# Charger le modèle YOLOv8 pour la détection des personnes
model = YOLO('yolov8n.pt')

# Initialisation de MediaPipe pour la détection des poses
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

POSITIONS = ["priere", "mains_a_gauche", "mains_a_droite", "mains_fesses", "main_en_air",
             "mains_ecartees"]
TARGET_POSE_TIME = 1.5


class ScoreTracker:
    def __init__(self):
        self.score = 0
        self.last_valid_pose = None
        self.pose_start_time = None

    def update_score(self, position, is_correct):
        current_time = time.time()

        # Valider la pose après un certain temps
        if is_correct and (self.last_valid_pose != position):
            if self.pose_start_time is None:
                self.pose_start_time = current_time

            # Vérifier si la pose est maintenue assez longtemps
            if current_time - self.pose_start_time >= TARGET_POSE_TIME:
                self.score += 100  # Augmenter le score
                self.last_valid_pose = position
                self.pose_start_time = None
                return True
        elif not is_correct or self.last_valid_pose == position:
            # Réinitialiser le temps si pose incorrecte
            self.pose_start_time = None

    def get_score(self):
        return int(self.score)


def is_good_position(landmarks, position):
    # [Votre code de vérification de pose existant]
    correct = True
    if position == "priere":
        # Mains jointes: vérifier si la distance entre les poignets est inférieure à un seuil
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        distance = np.sqrt((left_wrist.x - right_wrist.x) ** 2 + (left_wrist.y - right_wrist.y) ** 2)
        if distance > 0.3:
            correct = False


    elif position == "mains_a_droite":
        nose_x = landmarks[mp_pose.PoseLandmark.NOSE.value].x
        left_wrist_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x
        right_wrist_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x
        if left_wrist_x > nose_x or right_wrist_x > nose_x:
            correct = False


    elif position == "mains_tete":
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        if left_wrist_y > nose_y or right_wrist_y > nose_y:
            correct = False


    elif position == "mains_fesses":
        left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        if left_wrist_y == left_hip_y or right_wrist_y == right_hip_y:
            correct = False


    elif position == "mains_a_gauche":
        nose_x = landmarks[mp_pose.PoseLandmark.NOSE.value].x
        left_wrist_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x
        right_wrist_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x
        if left_wrist_x < nose_x or right_wrist_x < nose_x:
            correct = False

    elif position == "main_en_air":
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        if left_wrist_y > left_shoulder_y or right_wrist_y > right_shoulder_y:
            correct = False

    elif position == "mains_ecartees":
        # Vérifier si les bras sont écartés
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        shoulder_width = np.abs(left_shoulder.x - right_shoulder.x)
        wrist_distance = np.abs(left_wrist.x - right_wrist.x)

    # [Autres conditions de position similaires]
    return correct


# Fonction pour superposer l'image de la pose
def overlay_pose_with_transparency(frame, pose_image):
    # Assurez-vous que l'image de la pose a 4 canaux (R, G, B, A)
    if pose_image.shape[2] == 4:
        # Séparer les canaux : RGB et alpha
        img_rgb = pose_image[:, :, :3]
        img_alpha = pose_image[:, :, 3] / 255.0  # Normaliser l'alpha (0 à 1)

        # Définir la position où l'image de la pose sera affichée (coin inférieur droit)
        height, width, _ = frame.shape
        pose_height, pose_width, _ = img_rgb.shape
        x_offset = width - pose_width - 10
        y_offset = height - pose_height - 10

        # Superposer l'image de la pose avec transparence sur le frame
        for c in range(0, 3):  # Pour chaque canal (R, G, B)
            frame[y_offset:y_offset + pose_height, x_offset:x_offset + pose_width, c] = \
                (1. - img_alpha) * frame[y_offset:y_offset + pose_height, x_offset:x_offset + pose_width, c] + \
                img_alpha * img_rgb[:, :, c]
    else:
        # Si l'image de la pose n'a pas d'alpha (pas de transparence), on l'ajoute simplement au frame
        height, width, _ = frame.shape
        pose_height, pose_width, _ = pose_image.shape
        x_offset = width - pose_width - 10
        y_offset = height - pose_height - 10
        frame[y_offset:y_offset + pose_height, x_offset:x_offset + pose_width] = pose_image

    return frame


cap = cv2.VideoCapture(0)

score_tracker = ScoreTracker()
indice_pose = 0
last_change_time = time.time()

while cap.isOpened():


    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results[0].boxes

    # Charger l'image de la pose cible
    pose_image = cv2.imread(f"poses/{POSITIONS[indice_pose]}.png", cv2.IMREAD_UNCHANGED)
    frame = overlay_pose_with_transparency(frame, pose_image)

    for box in detections:
        if box.cls == 0:  # Classe 0 correspond à "person" dans YOLOv8
            xyxy = box.xyxy[0].cpu().numpy()  # Convertir en numpy array
            x1, y1, x2, y2 = map(int, xyxy)  # Convertir en entiers

            person_roi = frame[y1:y2, x1:x2]

            # Conversion de l'image en RGB pour MediaPipe
            person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)

            # Détection de pose avec MediaPipe
            results_pose = pose.process(person_rgb)

            cv2.putText(frame, POSITIONS[indice_pose], (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

            results = model(frame)
            detections = results[0].boxes



            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)

            person_roi = frame[y1:y2, x1:x2]
            person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(person_rgb)

            # Afficher informations
            cv2.putText(frame, POSITIONS[indice_pose], (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)
            score_text = f"Score: {score_tracker.get_score()}"
            cv2.putText(frame, score_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark

                is_correct = is_good_position(landmarks, POSITIONS[indice_pose])

                # Mettre à jour le score
                ok = score_tracker.update_score(POSITIONS[indice_pose], is_correct)
                if ok :
                    indice_pose = randint(0, len(POSITIONS) - 1)
                if is_correct:
                    cv2.putText(frame, "Pose correcte!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                cv2.LINE_AA)

                else:
                    cv2.putText(frame, "Essaye encore!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)

                # Dessiner les points clés de la pose
                mp.solutions.drawing_utils.draw_landmarks(person_roi, results_pose.pose_landmarks,
                                                          mp_pose.POSE_CONNECTIONS)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

    cv2.imshow("Just Dance", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()