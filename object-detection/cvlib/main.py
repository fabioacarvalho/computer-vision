import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2

# Reading input image - original
img = cv2.imread("img/image1.jpg")

# converting image in grayscale
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img_gray)

# bounding box, name of objetct and
bbox, label, conf = cv.detect_common_objects(img)

# drawing bounding box over detected objects at original image
out = draw_bbox(img, bbox, label, conf)

cv2.imwrite("img/image1_bbox.jpg", out)
cv2.destroyAllWindows()
