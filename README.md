# Face-Mask-Detection-using-MobilenetV2
Face mask detection using Pretrained __MobilenetV2__ using __TensorFlow 2.3__ and __Keras API__ with help of Opencv's __haarcascade__ to help with Facial recognition before feeding the data to the model. 
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
