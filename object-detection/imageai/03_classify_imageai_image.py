from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd()
prediction = ImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(execution_path, "model/resnet50-19c8e357.pth"))
prediction.loadModel()


predictions, probabilities_percent = prediction.classifyImage(
    os.path.join(execution_path, "img/image2.jpg"),
    result_count=5,
    
)

for indice in range(len(predictions)):
    print(predictions[indice] + " : " + probabilities_percent[indice])


# for eachPrediction, eachProbability in zip(predictions, probabilities_percent):
#     print(eachPrediction , " : " , eachProbability)
