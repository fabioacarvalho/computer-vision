# Using Image AI

Here we're gonna use this lib to detect objects and image segmentation.

> Doc.: https://imageai.readthedocs.io/en/latest/



## Environment

We need use version python required at documentation, in this case, Python 3.10.

> You can go to at python page and type /ftp/python such as: https://www.python.org/ftp/python/

Now we need install the follow libs:

- Install Dependencies (CPU):
    
    ```python
    pip install cython pillow>=7.0.0 numpy>=1.18.1 opencv-python>=4.1.2 torch>=1.9.0 --extra-index-url https://download.pytorch.org/whl/cpu torchvision>=0.10.0 --extra-index-url https://download.pytorch.org/whl/cpu pytest==7.1.3 tqdm==4.64.1 scipy>=1.7.3 matplotlib>=3.4.3 mock==4.0.3
    
    ```
    

- Install Dependencies (GPU/CUDA):
    
    ```python
    pip install cython pillow>=7.0.0 numpy>=1.18.1 opencv-python>=4.1.2 torch>=1.9.0 --extra-index-url https://download.pytorch.org/whl/cu102 torchvision>=0.10.0 --extra-index-url https://download.pytorch.org/whl/cu102 pytest==7.1.3 tqdm==4.64.1 scipy>=1.7.3 matplotlib>=3.4.3 mock==4.0.3
    ```
    

- If you plan to train custom AI models, run the command below to install an extra dependency:
    
    ```python
    pip install pycocotools@git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI
    ```
    

- ImageAI:
    
    ```python
    pip install imageai --upgrade
    ```

So now we need download the YOLO model and put into a folder called `model`.

---

## Detect object from image

Example project:

```python
from imageai.Detection import ObjectDetection
import os

path_exec = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(path_exec, "model/yolov3.pt"))  # Path where is your file
detector.loadModel()  # Load weight


detections = detector.detectObjectsFromImage(
    input_image=os.path.join(path_exec, "img/image1.jpg"),
    output_image_path=os.path.join(path_exec, "img/image1_detect.jpg"),
    minimum_percentage_probability=40
)


```

## Detect object from video

Example project:

```python
from imageai.Detection import VideoObjectDetection
import os

BASE_DIR = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(BASE_DIR, "model/yolov3.pt"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(
    input_file_path=os.path.join(BASE_DIR, "video/traffic.mp4"),
    output_file_path=os.path.join(BASE_DIR, "video/traffic_detected.mp4"),
    frames_per_second=20,
    log_progress=True
)

print(video_path)

```
