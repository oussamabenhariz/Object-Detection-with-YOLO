# Object Detection with TensorFlow and YOLO Algorithm using EfficientNet

## Overview

This project demonstrates object detection using the YOLO (You Only Look Once) algorithm with TensorFlow. The model is based on the EfficientNet architecture and is pretrained on the COCO dataset. The goal is to detect and classify objects among 20 classes, including aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, dining table, dog, horse, motorbike, person, potted plant, sheep, sofa, train, and TV monitor.

## Requirements

Make sure you have the following dependencies installed:

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Albumentations
- Opencv

You can install the required packages using the following command:

```bash
pip install tensorflow numpy matplotlib opencv-python albumentations
```
##Usage
Open and run the object detection notebook (yolo_algorithm_for_object_detection.ipynb) in Jupyter Notebook or JupyterLab.
Follow the instructions within the notebook, specifying the path to the image you want to test.
The notebook will display the image with bounding boxes and class labels drawn around detected objects. Additionally, the results will be saved in the results folder.

## Customization

If you want to use a different set of classes or fine-tune the model, you can modify the classes list in the notebook. You may also experiment with different YOLO versions or change the preprocessing steps based on your specific requirements.

## Acknowledgements
This project is built on top of the TensorFlow framework and utilizes the EfficientNet architecture and YOLO algorithm. Credits to the TensorFlow team and the authors of the EfficientNet and YOLO papers for their contributions.



## References

- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [TensorFlow](https://www.tensorflow.org/)
- [COCO Dataset](https://cocodataset.org/)
- [VOC DAtaset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

  
Feel free to contribute, report issues, or suggest improvements. Happy coding!
