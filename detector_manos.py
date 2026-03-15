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
def dedos_abiertos(hand):

    dedos = []

    # Pulgar
    if hand.landmark[4].x < hand.landmark[3].x:
        dedos.append(1)
    else:
        dedos.append(0)

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
# DETECTAR LETRAS LSM
# -----------------------------
def detectar_lsm(hand):

    dedos = dedos_abiertos(hand)

    muñeca = hand.landmark[0]
    base_medio = hand.landmark[9]

    pulgar = hand.landmark[4]
    indice = hand.landmark[8]

    tamano_mano = distancia(muñeca, base_medio)

    dist_pi = distancia(pulgar, indice)

    ratio = dist_pi / tamano_mano

    # -----------------------------
    # A
    if dedos == [1,0,0,0,0]:
        return "A"

    # -----------------------------
    # B
    if dedos == [0,1,1,1,1]:
        return "B"

    # -----------------------------
    # C
    if 0.4 < ratio < 0.8:
        return "C"

    # -----------------------------
    # D
    if dedos == [0,1,0,0,0]:
        return "D"

    # -----------------------------
    # E
    if dedos == [0,0,0,0,0]:
        return "E"

    # -----------------------------
    # F
    if distancia(hand.landmark[8], hand.landmark[4]) < tamano_mano * 0.25:
        if dedos[2] == 1 and dedos[3] == 1 and dedos[4] == 1:
            return "F"

    # -----------------------------
    # I
    menique = hand.landmark[20]
    base_menique = hand.landmark[18]

    if dedos == [0,0,0,0,1]:
        if menique.y < base_menique.y:  
            return "I"

    # -----------------------------
    # K
    indice = hand.landmark[8]
    medio = hand.landmark[12]
    pulgar = hand.landmark[4]

    if dedos == [1,1,1,0,0]:
        if pulgar.y > indice.y and pulgar.y > medio.y:
            return "K"

    # -----------------------------
    # L
    if dedos == [1,1,0,0,0]:
        return "L"
    
    # -----------------------------
    # M
    if dedos == [0,0,0,0,0]:

        if distancia(hand.landmark[8], hand.landmark[4]) < tamano_mano*0.25 and \
        distancia(hand.landmark[12], hand.landmark[4]) < tamano_mano*0.25 and \
        distancia(hand.landmark[16], hand.landmark[4]) < tamano_mano*0.25:
            return "M"
        
    # -----------------------------
    # N
    if dedos == [0,0,0,0,0]:

        if distancia(hand.landmark[8], hand.landmark[4]) < tamano_mano*0.25 and \
        distancia(hand.landmark[12], hand.landmark[4]) < tamano_mano*0.25:
            return "N"
        
    # -----------------------------
    # O
    if distancia(hand.landmark[8], hand.landmark[4]) < tamano_mano*0.3 and \
    distancia(hand.landmark[12], hand.landmark[4]) < tamano_mano*0.3 and \
    distancia(hand.landmark[16], hand.landmark[4]) < tamano_mano*0.3 and \
    distancia(hand.landmark[20], hand.landmark[4]) < tamano_mano*0.3:
        return "O"

# -----------------------------
# CAMARA
# -----------------------------
cap = cv2.VideoCapture(1)

cap.set(3, 640)
cap.set(4, 480)

texto = ""
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

            for hand in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                letra = detectar_lsm(hand)

                if letra == gesto_anterior:
                    contador += 1
                else:
                    contador = 0

                gesto_anterior = letra

                if contador > 5 and letra != "":
                    texto = letra

        cv2.putText(
            frame,
            f"LSM: {texto}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow("Traductor LSM", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break


cap.release()
cv2.destroyAllWindows()