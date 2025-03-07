from pixellib.torchbackend.instance import instanceSegmentation

inst = instanceSegmentation()
inst.load_model("model/pointrend_resnet50.pkl")
inst.segmentImage(
    "img/image1.jpg",
    show_bboxes=True,
    output_image_name="img/image1_seg.jpg"
)
