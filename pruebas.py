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

    if mano == "Right":
        dedos.append(1 if hand.landmark[4].x < hand.landmark[3].x else 0)
    else:
        dedos.append(1 if hand.landmark[4].x > hand.landmark[3].x else 0)

    dedos.append(1 if hand.landmark[8].y < hand.landmark[6].y else 0)
    dedos.append(1 if hand.landmark[12].y < hand.landmark[10].y else 0)
    dedos.append(1 if hand.landmark[16].y < hand.landmark[14].y else 0)
    dedos.append(1 if hand.landmark[20].y < hand.landmark[18].y else 0)

    return dedos

# -----------------------------
# DETECTAR LETRAS O-Z
# -----------------------------
def detectar_letra_izquierda(hand, mano):

    dedos = dedos_abiertos(hand, mano)

    muñeca = hand.landmark[0]
    base_medio = hand.landmark[9]

    pulgar = hand.landmark[4]
    indice = hand.landmark[8]
    medio = hand.landmark[12]
    anular = hand.landmark[16]
    meñique = hand.landmark[20]

    tamano_mano = distancia(muñeca, base_medio)

    ang_indice = angulo(hand.landmark[5], indice)
    ang_medio = angulo(base_medio, medio)

    dist_pi = distancia(pulgar, indice)
    ratio = dist_pi / tamano_mano

    dist_im = distancia(indice, medio) / tamano_mano

    # -----------------------------
    # LETRAS MANO IZQUIERDA O-Z
    # -----------------------------

    # O
    if dedos == [1,0,0,0,0]:
        return "O", ang_indice, ang_medio, ratio, dedos

    # P
    if dedos == [1,1,0,0,0] and -110 <= ang_indice <= -80:
        return "P", ang_indice, ang_medio, ratio, dedos

    # Q
    if dedos == [1,1,0,0,0] and 0.40 <= ratio <= 0.80:
        return "Q", ang_indice, ang_medio, ratio, dedos

    # R
    if dedos == [1,1,0,0,0] and -80 <= ang_indice <= -50:
        return "R", ang_indice, ang_medio, ratio, dedos

    # S
    if dedos == [0,0,0,0,0]:
        return "S", ang_indice, ang_medio, ratio, dedos

    # T
    if dedos == [1,1,0,0,0] and -10 <= ang_indice <= 10:
        return "T", ang_indice, ang_medio, ratio, dedos

    # U
    if dedos == [0,1,1,0,0] and dist_im < 0.18:
        return "U", ang_indice, ang_medio, ratio, dedos

    # V
    if dedos == [0,1,1,0,0] and dist_im >= 0.18:
        return "V", ang_indice, ang_medio, ratio, dedos

    # W
    if dedos == [0,1,1,1,0]:
        return "W", ang_indice, ang_medio, ratio, dedos

    # X
    if dedos == [0,1,0,0,0] and -80 <= ang_indice <= -60:
        return "X", ang_indice, ang_medio, ratio, dedos

    # Y
    if dedos == [0,1,0,0,1]:
        return "Y", ang_indice, ang_medio, ratio, dedos

    # Z
    if dedos == [1,1,1,0,0] and -80 <= ang_indice <= -50:
        return "Z", ang_indice, ang_medio, ratio, dedos

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

                # SOLO MANO IZQUIERDA
                if mano == "Left":

                    letra, ang_indice, ang_medio, ratio, dedos = detectar_letra_izquierda(hand, mano)

                    if letra != "":
                        letra_actual = letra

                    cv2.putText(frame, f"Letra: {letra_actual}", (10,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                    cv2.putText(frame, f"Mano: {mano}", (10,80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

                    cv2.putText(frame, f"Ang indice: {int(ang_indice)}", (10,110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

                    cv2.putText(frame, f"Ang medio: {int(ang_medio)}", (10,140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

                    cv2.putText(frame, f"Ratio PI: {ratio:.2f}", (10,170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

                    cv2.putText(frame, f"Dedos: {dedos}", (10,200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.imshow("Tester Mano Izquierda O-Z", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()