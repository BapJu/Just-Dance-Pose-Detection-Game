import time
import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

# Charger le modèle YOLOv8 pour la détection des personnes
model = YOLO('yolov8n.pt')

# Initialisation de MediaPipe pour la détection des poses
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

POSITIONS = ["priere", "mains_a_droite","mains_a_gauche","mains_ecartees", "mains_fesses", "main_en_air"]
POSE_TIME_LIMIT = 4  # Temps maximum pour effectuer une pose (en secondes)

# Fonction pour vérifier si la pose correspond à une pose cible
def is_good_position(landmarks, position):
    correct = True
    if position == "priere":
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        distance = np.sqrt((left_wrist.x - right_wrist.x) ** 2 + (left_wrist.y - right_wrist.y) ** 2)
        if distance > 0.3:  # Seuil à ajuster selon les besoins
            correct = False
    elif position == "mains_a_gauche":
        nose_x = landmarks[mp_pose.PoseLandmark.NOSE.value].x
        left_wrist_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x
        right_wrist_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x
        if left_wrist_x > nose_x or right_wrist_x > nose_x:
            correct = False
    elif position == "mains_a_droite":
        nose_x = landmarks[mp_pose.PoseLandmark.NOSE.value].x
        left_wrist_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x
        right_wrist_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x
        if left_wrist_x < nose_x or right_wrist_x < nose_x:
            correct = False
    elif position == "mains_fesses":
        left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        if left_wrist_y != left_hip_y or right_wrist_y != right_hip_y:
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

        if wrist_distance < 1.5 * shoulder_width:
            correct = False
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        # Vérifier la hauteur des genoux par rapport aux hanches
        if left_knee.y < left_hip.y - 0.1 or right_knee.y < right_hip.y - 0.1:  # Ajustez le seuil (0.1) si nécessaire
            correct = False

    return correct

# Initialisation de la capture vidéo
cap = cv2.VideoCapture(0)

indice_pose = 0
last_change_time = time.time()
pose_start_time = None  # Heure de début pour chronométrer chaque pose

while cap.isOpened():
    if indice_pose > len(POSITIONS) - 1:
        indice_pose = 0

    ret, frame = cap.read()
    if not ret:
        break

    # Utilisation de YOLOv8 pour détecter les personnes
    results = model(frame)
    detections = results[0].boxes

    # Filtrer les détections pour ne garder que les personnes
    for box in detections:
        if box.cls == 0:  # Classe 0 correspond à "person"
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            person_roi = frame[y1:y2, x1:x2]
            person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)

            # Détection de pose avec MediaPipe
            results_pose = pose.process(person_rgb)

            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark

                # Démarrer le chronomètre pour cette pose si ce n'est pas déjà fait
                if pose_start_time is None:
                    pose_start_time = time.time()

                # Temps restant pour effectuer la pose
                elapsed_time = time.time() - pose_start_time
                time_remaining = max(0, POSE_TIME_LIMIT - elapsed_time)

                if is_good_position(landmarks, POSITIONS[indice_pose]):
                    cv2.putText(frame, "Pose correcte!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if elapsed_time >= POSE_TIME_LIMIT:
                        indice_pose += 1
                        pose_start_time = None  # Réinitialiser le chronomètre
                else:
                    cv2.putText(frame, "Essaye encore!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Afficher le compteur
                cv2.putText(frame, f"Temps restant: {time_remaining:.1f}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 0), 2)

                # Si le temps est écoulé et la pose est incorrecte, réinitialiser
                if time_remaining == 0 and not is_good_position(landmarks, POSITIONS[indice_pose]):
                    pose_start_time = None  # Réinitialiser le chronomètre

            # Encadrer la personne détectée
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

    # Charger et afficher l'image de la pose cible
    pose_image = cv2.imread(f"poses/{POSITIONS[indice_pose]}.png", cv2.IMREAD_UNCHANGED)
    if pose_image is not None:
        height, width, _ = frame.shape
        pose_height, pose_width, _ = pose_image.shape
        x_offset = width - pose_width - 10
        y_offset = height - pose_height - 10
        frame[y_offset:y_offset + pose_height, x_offset:x_offset + pose_width] = pose_image[:, :, :3]

    # Afficher l'image avec les résultats
    cv2.imshow("Just Dance", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
