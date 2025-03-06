from imageai.Detection import ObjectDetection
import os

path_exec = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(path_exec, "model/yolov3.pt"))  # Path where is your file
detector.loadModel()  # Load weight


detections = detector.detectObjectsFromImage(
    input_image=os.path.join(path_exec, "img/image4.jpg"),
    output_image_path=os.path.join(path_exec, "img/image4_detect.jpg"),
    minimum_percentage_probability=50
)


