import time

import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

# Charger le modèle YOLOv8 pour la détection des personnes
model = YOLO('yolov8n.pt')  # Charger le modèle YOLOv8n pré-entrainé

# Initialisation de MediaPipe pour la détection des poses
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


POSITIONS = ["priere","mains_a_droite","mains_tete","mains_fesses","mains_fesses","main_en_air"]

# Fonction pour vérifier si la pose correspond à une pose cible (Dab)
def is_good_position(landmarks, position):
    correct = True


    if position == "priere" :
        # Mains jointes: vérifier si la distance entre les poignets est inférieure à un seuil
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        distance = np.sqrt((left_wrist.x - right_wrist.x) ** 2 + (left_wrist.y - right_wrist.y) ** 2)

        if distance > 0.3:  # Seuil à ajuster selon les besoins
            correct = False

    if position == "mains_a_droite" :
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

# Fonction pour dessiner la pose cible Dab sur l'écran


# Initialisation de la capture vidéo
cap = cv2.VideoCapture(0)



indice_pose = 0
delay = 3
last_change_time = time.time()

while cap.isOpened():

    if indice_pose> len(POSITIONS)-1 :
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

            cv2.putText(frame, POSITIONS[indice_pose], (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # Si des points clés sont détectés
            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark


                # Vérifier si la pose correspond
                if is_good_position(landmarks,POSITIONS[indice_pose]):
                    current_time = time.time()
                    cv2.putText(frame, "Pose correcte!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                cv2.LINE_AA)
                    if current_time - last_change_time >= delay:
                        # Affiche le message et mets à jour le dernier changement

                        last_change_time = current_time  # Met à jour l'heure de changement
                        indice_pose += 1  # Incrémente l'indice

                else:
                    cv2.putText(frame, "Essaye encore!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Dessiner les points clés de la pose
                mp.solutions.drawing_utils.draw_landmarks(person_roi, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Affichage des landmarks de la pose détectée (superposition)
                for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
                    x = int(landmark.x * person_roi.shape[1])
                    y = int(landmark.y * person_roi.shape[0])
                    cv2.circle(frame, (x + x1, y + y1), 5, (0, 255, 0), -1)  # Afficher les landmarks en vert

            # Encadrer la personne détectée
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

    # Afficher l'image avec les résultats
    cv2.imshow("Just Dance ", frame)

    # Sortir avec 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break



# Libérer les ressources
cap.release()
cv2.destroyAllWindows()

