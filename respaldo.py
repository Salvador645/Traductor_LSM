import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# -----------------------------
# DISTANCIA ENTRE LANDMARKS
# -----------------------------
def distancia(p1, p2):

    return math.sqrt(
        (p1.x - p2.x)**2 +
        (p1.y - p2.y)**2
    )


# -----------------------------
# ANGULO ENTRE DOS PUNTOS
# -----------------------------
def angulo(p1, p2):

    dx = p2.x - p1.x
    dy = p2.y - p1.y

    ang = math.degrees(math.atan2(dy, dx))

    return ang


# -----------------------------
# DEDOS ABIERTOS
# -----------------------------
def dedos_abiertos(hand, mano):

    dedos = []

    # pulgar
    if mano == "Right":
        dedos.append(1 if hand.landmark[4].x < hand.landmark[3].x else 0)
    else:
        dedos.append(1 if hand.landmark[4].x > hand.landmark[3].x else 0)

    # índice
    dedos.append(1 if hand.landmark[8].y < hand.landmark[6].y else 0)

    # medio
    dedos.append(1 if hand.landmark[12].y < hand.landmark[10].y else 0)

    # anular
    dedos.append(1 if hand.landmark[16].y < hand.landmark[14].y else 0)

    # meñique
    dedos.append(1 if hand.landmark[20].y < hand.landmark[18].y else 0)

    return dedos


# -----------------------------
# DETECTAR LETRA
# -----------------------------
def detectar_letra(hand, mano):

    dedos = dedos_abiertos(hand, mano)

    muñeca = hand.landmark[0]
    base_medio = hand.landmark[9]

    pulgar = hand.landmark[4]
    indice = hand.landmark[8]

    tamano_mano = distancia(muñeca, base_medio)

    # ANGULO DEL INDICE
    indice_base = hand.landmark[5]
    indice_tip = hand.landmark[8]

    ang_indice = angulo(indice_base, indice_tip)

    # ANGULO DEL MEDIO
    medio_base = hand.landmark[9]
    medio_tip = hand.landmark[12]

    ang_medio = angulo(medio_base, medio_tip)

    dist_pi = distancia(pulgar, indice)
    ratio = dist_pi / tamano_mano


    # -----------------------------
    # A
    # -----------------------------
    if dedos == [1,0,0,0,0]:
        return "A", ang_indice


    # -----------------------------
    # B
    # -----------------------------
    if dedos == [0,1,1,1,1]:

        if abs(ang_indice) > 60:
            return "B", ang_indice


    # -----------------------------
    # C
    # -----------------------------
    if 0.4 < ratio < 0.8:
        return "C", ang_indice


    # -----------------------------
    # D
    # -----------------------------
    if dedos == [0,1,0,0,0]:

        if abs(ang_indice) > 60:
            return "D", ang_indice


    # -----------------------------
    # E
    # -----------------------------
    if dedos == [0,0,0,0,0]:
        return "E", ang_indice


    # -----------------------------
    # F
    # -----------------------------
    if distancia(hand.landmark[8], hand.landmark[4]) < tamano_mano * 0.25:

        if dedos[2] and dedos[3] and dedos[4]:
            return "F", ang_indice


    # -----------------------------
    # G
    # -----------------------------
    if dedos == [1,1,0,0,0]:

        if abs(ang_indice) < 30 or abs(ang_indice) > 150:
            return "G", ang_indice


    # -----------------------------
    # H
    # -----------------------------
    if dedos == [0,1,1,0,0]:

        if abs(ang_indice) < 30 and abs(ang_medio) < 30:
            return "H", ang_indice


    # -----------------------------
    # I
    # -----------------------------
    if dedos == [0,0,0,0,1]:
        return "I", ang_indice


    # -----------------------------
    # L
    # -----------------------------
    if dedos == [1,1,0,0,0]:

        if abs(ang_indice) > 60:
            return "L", ang_indice


    return "", ang_indice


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

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                mano = results.multi_handedness[i].classification[0].label

                letra, ang_indice = detectar_letra(hand, mano)

                if letra != "":
                    letra_actual = letra


                # mostrar angulo
                cv2.putText(
                    frame,
                    f"Angulo indice: {int(ang_indice)}",
                    (10,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,0),
                    2
                )


        # mostrar letra
        cv2.putText(
            frame,
            f"Letra: {letra_actual}",
            (10,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow("Detector LSM", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break


cap.release()
cv2.destroyAllWindows()