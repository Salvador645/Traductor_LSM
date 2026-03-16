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
# DETECTOR MANO DERECHA (A-N)
# -----------------------------
def detectar_letra_derecha(hand, mano):
    dedos = dedos_abiertos(hand, mano)
    muñeca = hand.landmark[0]
    base_medio = hand.landmark[9]
    pulgar = hand.landmark[4]
    indice = hand.landmark[8]
    medio = hand.landmark[12]

    tamano_mano = distancia(muñeca, base_medio)
    ang_indice = angulo(hand.landmark[5], indice)
    ang_medio = angulo(base_medio, medio)
    dist_pi = distancia(pulgar, indice)
    ratio = dist_pi / tamano_mano
    dist_im = distancia(indice, medio) / tamano_mano

    if dedos == [1,0,0,0,0]: return "A", ang_indice, ang_medio, ratio, dedos
    if dedos == [0,1,1,1,1]: return "B", ang_indice, ang_medio, ratio, dedos
    if 0.4 < ratio < 0.8 and dedos[1:] == [1,1,1,1]: return "C", ang_indice, ang_medio, ratio, dedos
    if dedos == [0,1,0,0,0] and 0.7 < ratio < 0.95: return "D", ang_indice, ang_medio, ratio, dedos
    if dedos == [0,0,0,0,0]: return "E", ang_indice, ang_medio, ratio, dedos
    if dedos == [1,0,1,1,1] and 0.20 <= ratio <= 0.25: return "F", ang_indice, ang_medio, ratio, dedos
    if dedos == [1,1,0,0,0] and -75 <= ang_indice <= -55: return "G", ang_indice, ang_medio, ratio, dedos
    if dedos == [1,1,1,0,0] and -75 <= ang_indice <= -55: return "H", ang_indice, ang_medio, ratio, dedos
    if dedos == [0,0,0,0,1]: return "I", ang_indice, ang_medio, ratio, dedos
    if dedos == [1,0,0,0,1]: return "J", ang_indice, ang_medio, ratio, dedos
    if dedos == [0,1,0,0,1]: return "K", ang_indice, ang_medio, ratio, dedos
    if dedos == [1,1,0,0,0] and -95 <= ang_indice <= -85: return "L", ang_indice, ang_medio, ratio, dedos
    if dedos == [0,1,1,1,0]: return "M", ang_indice, ang_medio, ratio, dedos
    if dedos == [0,1,1,0,0] and -105 <= ang_indice <= -75 and dist_im < 0.18: return "N", ang_indice, ang_medio, ratio, dedos

    return "", ang_indice, ang_medio, ratio, dedos

# -----------------------------
# DETECTOR MANO IZQUIERDA (O-Z)
# -----------------------------
def detectar_letra_izquierda(hand, mano):
    dedos = dedos_abiertos(hand, mano)
    muñeca = hand.landmark[0]
    base_medio = hand.landmark[9]
    pulgar = hand.landmark[4]
    indice = hand.landmark[8]
    medio = hand.landmark[12]

    tamano_mano = distancia(muñeca, base_medio)
    ang_indice = angulo(hand.landmark[5], indice)
    ang_medio = angulo(base_medio, medio)
    dist_pi = distancia(pulgar, indice)
    ratio = dist_pi / tamano_mano
    dist_im = distancia(indice, medio) / tamano_mano

    if dedos == [1,0,0,0,0]: return "O", ang_indice, ang_medio, ratio, dedos
    if dedos == [1,1,0,0,0] and -110 <= ang_indice <= -80: return "P", ang_indice, ang_medio, ratio, dedos
    if dedos == [1,1,0,0,0] and 0.40 <= ratio <= 0.80: return "Q", ang_indice, ang_medio, ratio, dedos
    if dedos == [1,1,0,0,0] and -80 <= ang_indice <= -50: return "R", ang_indice, ang_medio, ratio, dedos
    if dedos == [0,0,0,0,0]: return "S", ang_indice, ang_medio, ratio, dedos
    if dedos == [1,1,0,0,0] and -10 <= ang_indice <= 10: return "T", ang_indice, ang_medio, ratio, dedos
    if dedos == [0,1,1,0,0] and dist_im < 0.18: return "U", ang_indice, ang_medio, ratio, dedos
    if dedos == [0,1,1,0,0] and dist_im >= 0.18: return "V", ang_indice, ang_medio, ratio, dedos
    if dedos == [0,1,1,1,0]: return "W", ang_indice, ang_medio, ratio, dedos
    if dedos == [0,1,0,0,0] and -80 <= ang_indice <= -60: return "X", ang_indice, ang_medio, ratio, dedos
    if dedos == [0,1,0,0,1]: return "Y", ang_indice, ang_medio, ratio, dedos
    if dedos == [1,1,1,0,0] and -80 <= ang_indice <= -50: return "Z", ang_indice, ang_medio, ratio, dedos

    return "", ang_indice, ang_medio, ratio, dedos

# -----------------------------
# CAMARA
# -----------------------------
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(3,1280)
cap.set(4,720)

letra_actual = ""
palabra = ""
historial = []

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        letra_detectada = ""

        if results.multi_hand_landmarks:
            for i, hand in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                mano = results.multi_handedness[i].classification[0].label

                if mano == "Right":
                    letra, ang_indice, ang_medio, ratio, dedos = detectar_letra_derecha(hand, mano)
                else:
                    letra, ang_indice, ang_medio, ratio, dedos = detectar_letra_izquierda(hand, mano)

                if letra != "":
                    letra_detectada = letra

                # Debug visual
                cv2.putText(frame, f"Mano: {mano}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.putText(frame, f"Ang indice: {int(ang_indice)}", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                cv2.putText(frame, f"Ang medio: {int(ang_medio)}", (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                cv2.putText(frame, f"Ratio PI: {ratio:.2f}", (10,170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                cv2.putText(frame, f"Dedos: {dedos}", (10,200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        # -----------------------------
        # HISTORIAL DE LETRAS PARA CAMBIO RAPIDO
        # -----------------------------
        historial.append(letra_detectada)
        if len(historial) > 5:
            historial.pop(0)
        # letra más frecuente en los últimos 5 frames
        letras_filtradas = [l for l in historial if l != ""]
        if letras_filtradas:
            letra_actual = max(set(letras_filtradas), key=letras_filtradas.count)

        cv2.putText(frame, f"Letra: {letra_actual}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Palabra: {palabra}", (10,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow("Interprete Lenguaje", frame)

        # -----------------------------
        # CONTROL CON TECLADO
        # -----------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER → agregar letra a palabra
            if letra_actual != "":
                palabra += letra_actual
        elif key == 32:  # SPACE → espacio
            palabra += " "
        elif key == 8:   # BACKSPACE → borrar última letra
            palabra = palabra[:-1]
        elif key == 27:  # ESC → salir
            break

cap.release()
cv2.destroyAllWindows()