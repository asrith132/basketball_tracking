import torch
from PIL import Image
import cv2

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img)

        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det
            if conf > 0.5 and int(cls) == 0:  # Only consider basketball detections
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, 'basketball', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = 'path/to/your/video.mp4'
    process_video(video_path)
