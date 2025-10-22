import time
from ultralytics import YOLO
import cv2

face_detector = YOLO("models/yolov11s-face.pt")
model = YOLO("runs/classify/ddd_default_s/weights/best.pt")

start_time = None  # se inicializa vacío
padding = 0 # padding around detected face

drowsiness_window = 5.0  # segundos para considerar somnolencia
drowsy_start = None

for frame_result in face_detector.predict(source=0, stream=True, device="cpu"):
    # Registrar el instante del primer frame
    if start_time is None:
        start_time = time.time()

    # Calcular segundos transcurridos desde el primer frame
    elapsed = time.time() - start_time

    frame = frame_result.orig_img.copy()

    for box in frame_result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - padding * w))
        y1 = max(0, int(y1 - padding * h))
        x2 = min(frame.shape[1], int(x2 + padding * w))
        y2 = min(frame.shape[0], int(y2 + padding * h))
        cropped_face = frame[y1:y2, x1:x2]

        if cropped_face.size == 0:
            continue

        results = model.predict(source=cropped_face, device="cpu", verbose=False)
        label = results[0].names[results[0].probs.top1]
        conf = results[0].probs.top1conf.item()

        if label == "drowsy" and conf > 0.7:
            if drowsy_start is None:
                drowsy_start = elapsed
            elif elapsed - drowsy_start >= drowsiness_window:
                cv2.putText(frame, "DROWSINESS DETECTED!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            drowsy_start = None

        # Dibujar rectángulo y etiqueta
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Mostrar el tiempo en el vídeo
    cv2.putText(frame, f"Elapsed: {elapsed:.1f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
