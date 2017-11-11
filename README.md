# ResNet for Sketch Features
## 1. Environment configuration
> - Ubuntu 14.04.5 LTS
> - Python 2.7.6
> - Tensorflow 1.4.0-rc1
## 2. Code:
TFRecord generation:
> - DataGenTFRecord.py From raw data generation TFRecord

Read data:
> - cifar_input.py : cifar10 or cifar100 input
> - sketchy_input.py sketchy input

Model build:
> - resnet_model.py : build ResNet Model

Train Model:
> - resnet_main.py : train model

## 3. Data:
Sketchy dataset

## 4. Run:
### 1). Generation TFRecord:
> - python DataGenTFRecord.py
### 2). Train Model:
> - ./run.sh

## 5. References:


![](https://raw.githubusercontent.com/BasicCoder/ResNetforSketchFeatures/master/5-160ZF94411.jpg)
**Note:**
> - The picture has nothing to do with the content!
