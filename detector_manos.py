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
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# -----------------------------
# DETECTAR DEDOS ABIERTOS
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
# DETECTAR LETRAS
# -----------------------------
def detectar_letra(hand, mano):

    dedos = dedos_abiertos(hand, mano)

    muñeca = hand.landmark[0]
    base_medio = hand.landmark[9]

    pulgar = hand.landmark[4]
    indice = hand.landmark[8]

    tamano_mano = distancia(muñeca, base_medio)
    dist_pi = distancia(pulgar, indice)

    ratio = dist_pi / tamano_mano


    # A
    if dedos == [1,0,0,0,0]:
        return "A"

    # B
    if dedos == [0,1,1,1,1]:
        return "B"

    # C
    if 0.4 < ratio < 0.8:
        return "C"

    # D
    if dedos == [0,1,0,0,0]:
        return "D"

    # E
    if dedos == [0,0,0,0,0]:
        return "E"

    # F
    if distancia(hand.landmark[8], hand.landmark[4]) < tamano_mano * 0.25:
        if dedos[2] == 1 and dedos[3] == 1 and dedos[4] == 1:
            return "F"

    # I
    if dedos == [0,0,0,0,1]:
        return "I"

    # L
    if dedos == [1,1,0,0,0]:

        dx = abs(hand.landmark[8].x - hand.landmark[6].x)
        dy = abs(hand.landmark[8].y - hand.landmark[6].y)

        if dy > dx:
            return "L"

    return ""


# -----------------------------
# CAMARA
# -----------------------------
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cap.set(3,1280)
cap.set(4,720)

letra_actual = ""
palabra = ""

gesto_anterior = ""
contador = 0


with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as hands:

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:

            for i, hand in enumerate(results.multi_hand_landmarks):

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                mano = results.multi_handedness[i].classification[0].label

                letra = detectar_letra(hand, mano)

                if letra == gesto_anterior:
                    contador += 1
                else:
                    contador = 1

                gesto_anterior = letra

                if contador > 10 and letra != "":
                    letra_actual = letra


        # -----------------------------
        # TEXTO EN PANTALLA
        # -----------------------------
        cv2.putText(
            frame,
            f"Letra: {letra_actual}",
            (10,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Palabra: {palabra}",
            (10,100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,255),
            2
        )


        cv2.imshow("Traductor LSM", frame)

        # -----------------------------
        # CONTROL CON TECLADO
        # -----------------------------
        key = cv2.waitKey(1) & 0xFF

        # ENTER → agregar letra
        if key == 13:
            if letra_actual != "":
                palabra += letra_actual

        # SPACE → espacio
        elif key == 32:
            palabra += " "

        # BACKSPACE → borrar
        elif key == 8:
            palabra = palabra[:-1]

        # ESC → salir
        elif key == 27:
            break


cap.release()
cv2.destroyAllWindows()