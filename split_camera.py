# import cv2
# import numpy as np

# # pre-trained ResNet SSD model for face detection
# net = cv2.dnn.readNetFromCaffe(
#     'deploy.prototxt', 
#     'res10_300x300_ssd_iter_140000.caffemodel'
# )

# cap = cv2.VideoCapture(0)
# # cap = cv2.VideoCapture("https://10.25.255.189:8080/video")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame = cv2.flip(frame, 1)

#     h, w = frame.shape[:2]
#     left_half = frame[:, :w // 2]

#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()

#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.5:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             text = f"{confidence:.2f}"
#             y = startY - 10 if startY - 10 > 10 else startY + 10
#             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
#             cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

#     # frame = frame[:h/2][:w/2]
#     cv2.imshow("Face Detection (ResNet-SSD)", left_half)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np

net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt', 
    'res10_300x300_ssd_iter_140000.caffemodel'
)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    left_half = frame[:, :w // 2]
    right_half = frame[:, w // 2:]

    blob_left = cv2.dnn.blobFromImage(left_half, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob_left)
    detections_left = net.forward()

    for i in range(detections_left.shape[2]):
        confidence = detections_left[0, 0, i, 2]
        if confidence > 0.5:
            box = detections_left[0, 0, i, 3:7] * np.array([w // 2, h, w // 2, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = f"{confidence:.2f}"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(left_half, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(left_half, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    blob_right = cv2.dnn.blobFromImage(right_half, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob_right)
    detections_right = net.forward()

    for i in range(detections_right.shape[2]):
        confidence = detections_right[0, 0, i, 2]
        if confidence > 0.5:
            box = detections_right[0, 0, i, 3:7] * np.array([w // 2, h, w // 2, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = f"{confidence:.2f}"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(right_half, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(right_half, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    cv2.imshow("left", left_half)
    cv2.imshow("right", right_half)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
