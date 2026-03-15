import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# DISTANCIA ENTRE LANDMARKS
# -----------------------------
def distancia(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# -----------------------------
# ANGULO ENTRE DOS PUNTOS
# -----------------------------
def angulo(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return math.degrees(math.atan2(dy, dx))

# -----------------------------
# DEDOS ABIERTOS
# -----------------------------
def dedos_abiertos(hand, mano):
    dedos = []
    # Pulgar
    if mano == "Right":
        dedos.append(1 if hand.landmark[4].x < hand.landmark[3].x else 0)
    else:
        dedos.append(1 if hand.landmark[4].x > hand.landmark[3].x else 0)
    # Índice
    dedos.append(1 if hand.landmark[8].y < hand.landmark[6].y else 0)
    # Medio
    dedos.append(1 if hand.landmark[12].y < hand.landmark[10].y else 0)
    # Anular
    dedos.append(1 if hand.landmark[16].y < hand.landmark[14].y else 0)
    # Meñique
    dedos.append(1 if hand.landmark[20].y < hand.landmark[18].y else 0)
    return dedos

# -----------------------------
# DETECTAR LETRA (VERSION DEPURADA)
# -----------------------------
def detectar_letra(hand, mano):
    dedos = dedos_abiertos(hand, mano)
    muñeca = hand.landmark[0]
    base_medio = hand.landmark[9]
    pulgar = hand.landmark[4]
    indice = hand.landmark[8]
    medio = hand.landmark[12]
    anular = hand.landmark[16]
    meñique = hand.landmark[20]
    tamano_mano = distancia(muñeca, base_medio)

    # ANGULOS
    ang_indice = angulo(hand.landmark[5], indice)
    ang_medio = angulo(base_medio, medio)
    ang_menique = angulo(hand.landmark[17], meñique)

    # RATIO pulgar-indice
    dist_pi = distancia(pulgar, indice)
    ratio = dist_pi / tamano_mano

    # Distancia índice-medio para U/V
    dist_im = distancia(indice, medio) / tamano_mano

    # -----------------------------
    # REGLAS DEPURADAS LETRAS FUNCIONALES
    # -----------------------------

    # A: Solo pulgar abierto
    if dedos == [1,0,0,0,0]:
        return "A", ang_indice, ang_medio, ratio, dedos

    # B: Todos los dedos excepto pulgar abiertos
    if dedos == [0,1,1,1,1]:
        return "B", ang_indice, ang_medio, ratio, dedos

    # C: Mano abierta en "C"
    if ratio > 0.4 and ratio < 0.8 and dedos[1:] == [1,1,1,1]:
        return "C", ang_indice, ang_medio, ratio, dedos

    # D: Índice extendido, resto cerrado
    if dedos == [0,1,0,0,0]:
        if 0.7 < ratio < 0.95:
            return "D", ang_indice, ang_medio, ratio, dedos

    # E: Todos los dedos cerrados
    if dedos == [0,0,0,0,0]:
        return "E", ang_indice, ang_medio, ratio, dedos

    
    # I: Solo meñique levantado
    if dedos == [0,0,0,0,1]:
        return "I", ang_indice, ang_medio, ratio, dedos

    # L: Pulgar e índice extendidos
    if dedos == [1,1,0,0,0]:
        return "L", ang_indice, ang_medio, ratio, dedos

    # U / V: Índice y medio levantados
    if dedos == [0,1,1,0,0]:
        if dist_im < 0.2:
            return "U", ang_indice, ang_medio, ratio, dedos
        else:
            return "V", ang_indice, ang_medio, ratio, dedos

    # W: Índice, medio y anular levantados
    if dedos == [0,1,1,1,0]:
        return "W", ang_indice, ang_medio, ratio, dedos
    
    # F: pulgar levantado, índice doblado tocando pulgar, medio/anular/meñique levantados
    if dedos == [1,0,1,1,1] and 0.20 <= ratio <= 0.25:
        return "F", ang_indice, ang_medio, ratio, dedos

    # Default: no detectada
    return "", ang_indice, ang_medio, ratio, dedos

# -----------------------------
# CAMARA
# -----------------------------
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(3,1280)
cap.set(4,720)

letra_actual = ""

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for i, hand in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                mano = results.multi_handedness[i].classification[0].label

                letra, ang_indice, ang_medio, ratio, dedos = detectar_letra(hand, mano)
                if letra != "":
                    letra_actual = letra

                # -----------------------------
                # DEBUG VISUAL
                # -----------------------------
                cv2.putText(frame, f"Letra: {letra_actual}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, f"Mano: {mano}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.putText(frame, f"Ang indice: {int(ang_indice)}", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                cv2.putText(frame, f"Ang medio: {int(ang_medio)}", (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                cv2.putText(frame, f"Ratio PI: {ratio:.2f}", (10,170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                cv2.putText(frame, f"Dedos: {dedos}", (10,200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.imshow("Detector LSM", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()