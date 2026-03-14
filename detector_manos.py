import cv2
import mediapipe as mp

# Inicializar mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# -----------------------------
# FUNCION PARA DETECTAR DEDOS
# -----------------------------
def dedos_abiertos(hand_landmarks):

    dedos = []

    # Pulgar
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        dedos.append(1)
    else:
        dedos.append(0)

    # Índice
    if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
        dedos.append(1)
    else:
        dedos.append(0)

    # Medio
    if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
        dedos.append(1)
    else:
        dedos.append(0)

    # Anular
    if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:
        dedos.append(1)
    else:
        dedos.append(0)

    # Meñique
    if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:
        dedos.append(1)
    else:
        dedos.append(0)

    return dedos


# -----------------------------
# CAMARA
# -----------------------------
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7) as hands:

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        # Voltear imagen para efecto espejo
        frame = cv2.flip(frame, 1)

        # Convertir a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar imagen
        results = hands.process(frame_rgb)

        texto = ""

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                # Dibujar mano
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Numerar puntos
                for id, lm in enumerate(hand_landmarks.landmark):

                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    cv2.putText(frame, str(id),
                                (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0,255,0), 1)

                # Detectar dedos
                dedos = dedos_abiertos(hand_landmarks)

                # -----------------
                # DETECTAR GESTOS
                # -----------------

                if dedos == [1,1,1,1,1]:
                    texto = "MANO ABIERTA"

                elif dedos == [0,0,0,0,0]:
                    texto = "PUNO"

                elif dedos == [1,0,0,0,0]:
                    texto = "PULGAR ARRIBA"

                else:
                    texto = "GESTO DESCONOCIDO"

        # Mostrar texto
        cv2.putText(frame, texto,
                    (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        # Mostrar ventana
        cv2.imshow("Detector de gestos", frame)

        # Salir con ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break


cap.release()
cv2.destroyAllWindows()