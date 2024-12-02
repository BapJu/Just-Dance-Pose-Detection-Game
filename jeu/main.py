import time
import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
from pydub import AudioSegment
from pydub.playback import play

# Charger le modèle YOLOv8 pour la détection des personnes
model = YOLO('yolov8n.pt')  # Charger le modèle YOLOv8n pré-entrainé

# Initialisation de MediaPipe pour la détection des poses
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

POSITIONS = ["priere", "mains_a_droite", "mains_fesses", "mains_fesses", "main_en_air"]
TARGET_POSE_TIME = 1.5  # Temps cible pour maintenir une pose
MAX_POSE_TIME_BONUS = 0.8  # Temps maximum pour obtenir un bonus de vitesse


song = AudioSegment.from_mp3("C:/dev/IAYolo/jeu/music/waka.mp3")
play(song)

# Fonction pour vérifier si la pose correspond à une pose cible (Dab)
def is_good_position(landmarks, position):
    correct = True
    if position == "priere":
        # Mains jointes: vérifier si la distance entre les poignets est inférieure à un seuil
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        distance = np.sqrt((left_wrist.x - right_wrist.x) ** 2 + (left_wrist.y - right_wrist.y) ** 2)
        if distance > 0.3:  # Seuil à ajuster selon les besoins
            correct = False

    if position == "mains_a_droite":
        nose_x = landmarks[mp_pose.PoseLandmark.NOSE.value].x
        left_wrist_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x
        right_wrist_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x
        if left_wrist_x > nose_x or right_wrist_x > nose_x:
            correct = False

    if position == "mains_tete":
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        if left_wrist_y > nose_y or right_wrist_y > nose_y:
            correct = False

    if position == "mains_fesses":
        left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        if left_wrist_y == left_hip_y or right_wrist_y == right_hip_y:
            correct = False

    if position == "main_en_air":
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        if left_wrist_y > left_shoulder_y or right_wrist_y > right_shoulder_y:
            correct = False

    return correct


# Fonction pour superposer l'image de la pose avec transparence
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


# Initialisation de la capture vidéo
cap = cv2.VideoCapture(0)

indice_pose = 0
last_change_time = time.time()

# Nouveaux compteurs pour le score
current_pose_start_time = None
pose_scores = [0] * len(POSITIONS)
total_score = 0

while cap.isOpened():
    if indice_pose >= len(POSITIONS) - 1:
        indice_pose = 0

    ret, frame = cap.read()
    if not ret:
        break

    # Utilisation de YOLOv8 pour détecter les personnes
    results = model(frame)  # Appliquer le modèle YOLOv8 sur l'image

    # Résultats de détection YOLOv8 sous forme de liste d'objets Box
    detections = results[0].boxes  # Récupérer les boîtes englobantes des objets détectés

    # Filtrer les détections pour ne garder que les personnes
    for box in detections:
        if box.cls == 0:  # Classe 0 correspond à "person" dans YOLOv8
            xyxy = box.xyxy[0].cpu().numpy()  # Convertir en numpy array
            x1, y1, x2, y2 = map(int, xyxy)  # Convertir en entiers

            person_roi = frame[y1:y2, x1:x2]

            # Conversion de l'image en RGB pour MediaPipe
            person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)

            # Détection de pose avec MediaPipe
            results_pose = pose.process(person_rgb)

            # Afficher la pose courante
            cv2.putText(frame, POSITIONS[indice_pose], (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # Afficher le score total
            cv2.putText(frame, f"Score Total: {total_score:.1f}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Si des points clés sont détectés
            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark

                # Vérifier si la pose correspond
                if is_good_position(landmarks, POSITIONS[indice_pose]):
                    current_time = time.time()
                    cv2.putText(frame, "Pose correcte!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                cv2.LINE_AA)

                    # Initialiser le temps de début de la pose si ce n'est pas déjà fait
                    if current_pose_start_time is None:
                        current_pose_start_time = current_time

                    # Calculer le temps passé dans la pose
                    pose_duration = current_time - current_pose_start_time

                    # Calculer le score pour cette pose
                    if pose_duration >= TARGET_POSE_TIME:
                        # Bonus de temps si pose maintenue au-delà du temps cible
                        bonus_time = min(pose_duration - TARGET_POSE_TIME, MAX_POSE_TIME_BONUS)

                        # Calculer le score de vitesse
                        if current_pose_start_time is not None:
                            speed_score = max(0, TARGET_POSE_TIME - (current_time - current_pose_start_time))

                            # Mettre à jour le score total
                            pose_score = speed_score + bonus_time
                            total_score += pose_score*10
                            pose_scores[indice_pose] = pose_duration

                        # Afficher le score de la pose
                        cv2.putText(frame, f"Pose Score: +{pose_score:.1f}", (50, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Vérifier si assez de temps est passé pour changer de pose
                        if pose_duration >= TARGET_POSE_TIME:
                            indice_pose += 1  # Incrémente l'indice
                            current_pose_start_time = None  # Réinitialiser le temps de début de pose

                else:
                    cv2.putText(frame, "Essaye encore!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    # Réinitialiser le temps de début de pose si la pose n'est plus correcte
                    current_pose_start_time = None

                # Dessiner les points clés de la pose
                mp.solutions.drawing_utils.draw_landmarks(person_roi, results_pose.pose_landmarks,
                                                          mp_pose.POSE_CONNECTIONS)

                # Affichage des landmarks de la pose détectée (superposition)
                for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
                    x = int(landmark.x * person_roi.shape[1])
                    y = int(landmark.y * person_roi.shape[0])
                    cv2.circle(frame, (x + x1, y + y1), 5, (0, 255, 0), -1)  # Afficher les landmarks en vert

            # Encadrer la personne détectée
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

    # Charger l'image de la pose cible (image PNG)
    pose_image = cv2.imread(f"poses/{POSITIONS[indice_pose]}.png", cv2.IMREAD_UNCHANGED)

    # Afficher l'image de la pose à réaliser avec transparence
    if pose_image is not None:
        frame = overlay_pose_with_transparency(frame, pose_image)

    # Afficher l'image avec les résultats
    cv2.imshow("Just Dance", frame)

    # Sortir avec 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        p.stop()
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()

