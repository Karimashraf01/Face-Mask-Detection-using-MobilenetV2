# Face-Mask-Detection-using-MobilenetV2
Face mask detection using Pretrained __MobilenetV2__ using __TensorFlow 2.3__ and __Keras API__ with help of Opencv's __haarcascade__ to help with Facial recognition before feeding the data to the model. 

<p align="center">
  <img  src="https://github.com/Karimashraf01/Face-Mask-Detection-using-MobilenetV2/blob/master/img_readme/test_img.jpg">
</p>

## Getting Started
### Prerequisites
- Python 3.6 or Higher
- TensorFlow 2.x
- opencv
- numpy

### Installing
This command will help to install the required independenies:
```
pip install -r requirements.txt
```

### Nvidia GPU acceleration
To use your __GPU__ in computing, You must install the required version of __CUDA Toolkit__ suitable with the version of your installed __Tensorflow__
for more help installing __CUDA Toolkit__ visit [Tensorflow site](https://www.tensorflow.org/install/gpu).

## Dataset
The dataset came form __Kaggle__ and had __12K examples__ splited into __2 class__:
- Wearing Mask.
- Not Wearing Mask.

Which was split to 3 __categories__:
- Training 
- Testing 
- Validation

To check the dataset from __Kaggle__ you can visit this [link](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset) .

### Dataset Sample:
<p align="center">
  <img  src="https://github.com/Karimashraf01/Face-Mask-Detection-using-MobilenetV2/blob/master/img_readme/sample.jpg">
</p>

## Model Architecture
The model consist of the pretrained network of __MobileNetV2__ and adding two layers:
- Flatten Layer 
- Dense layer with 2 neuron; one for each class with activation function of __Softmax_ 

The model Takes __RGB__ images with shape of __(1,128,128,3)__ and gives out An vector of two Values, The First value indicates the confidence of the predication of wearing a mask. And the second Value indicates the confidence of not wearing a mask.

Due to using __Softmax__ Function the sum of the two Values equal to one.
### MobileNetV2
<p align="center">
  <img  src="https://github.com/Karimashraf01/Face-Mask-Detection-using-MobilenetV2/blob/master/img_readme/mobilenetv2.jpg">
</p>

### Model Summary
<p align="center">
  <img  src="https://github.com/Karimashraf01/Face-Mask-Detection-using-MobilenetV2/blob/master/img_readme/Summary.jpg">
</p>

## Running The Model using Webcam
To run the model using __Webcam__ and highlighting the faces with a __Bounding Box__ indicating the Prediction run this command
```
python "WebCam.py"
```
### Prediction making
`WebCam.py` script uses opencv's __HaarCascade__ to cut the faces from the input frames from the webcam. And feeds the cropped image to the model to make a prediction and based on it a __Bounding Box__ with the prediction will be drawn on the output.
To Terminate the program press __ESC__.

