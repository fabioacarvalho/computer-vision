from imageai.Detection import VideoObjectDetection
import os

BASE_DIR = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(BASE_DIR, "model/yolov3.pt"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(
    input_file_path=os.path.join(BASE_DIR, "video/traffic.mp4"),
    output_file_path=os.path.join(BASE_DIR, "video/traffic_detected"),
    frames_per_second=20,
    log_progress=True
)

print(video_path)

