# CarND-Capstone Writeup

## Overview

The main goal of this project was to implement couple of ROS packages that all together can simulate simplified self-driving car components architecture. The packages enables to steer and drive autonomously a car within the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases) and the real Udacity car Carla.

## Walk-through Code

Majority of the code for the ROS packages were derived from the walk-through available with project instructions.

## First Project Run

This step causes a lot of problems for many Self-Driving Car Nanodegree students. The proposed architecture of the ROS packages and the Udacity simulator can only be run without any issues on modern powerful machines. If you run this project on older computer it will most likely not work out of the box. Hopefully I have found out single generic solution that enables to run this project on weaker machines not equipped with nVidia graphic cards with CUDA cores. You can even run this project in distributed environment e.g. ROS part on nVidia Jetson Nano/TX1/TX2 and simulator on the Intel x86(_64) PC. The setup description, tips and tricks deservers separate article on e.g. http://medium.com website.

## Traffic Light Detector

The walk-through code was missing implementation of Traffic Light Detector . There are plenty of options to implement it. For sure there are better solutions out there then the one described here. The training material suggested to use SSD (Single Shot Detector) for Traffic Light Detector. I decided to use YOLO (You Only Look Once) instead based on the quick comparison available at [article](https://technostacks.com/blog/yolo-vs-ssd/). I used the quicker solution fearing that SSD would overload my equipment. In addition I thought [Keras](https://keras.io/) would save me a little bit of development effort. I found https://github.com/qqwweee/keras-yolo3 repository that I used as a starting point. It turned out that indeed the repository includes plenty of useful code but unfortunately it does not meet 100% of requirements for this project. Eventually I came up with my own fork https://github.com/marcin-sielski/keras-yolo3 where I added useful tools and corrected some bugs. The repository became sub-module of this project. YOLO itself is not enough to to implement full Traffic Light Detector because it is still required to distinguish RED, YELLOW, GREEN lights. This is the reason there is a need to use additional component called Traffic Light Classifier. There are several ways to implement Traffic Light Classifier but I decided to use LeNet network which would take as an input bounding box images from YOLO and return one hot encoded light classification on the output.

### Traffic Light Classifier

1. Install additional dependencies.

    ```bash
    sudo apt install python-sklearn
    ```

2. Download data-set that includes images from camera mounted on the car within Udacity Simulator form https://drive.google.com/file/d/0Bw5abyXVejvMci03bFRueWVXX1U (note that Carla setup you would have to use real images instead).

3. Unzip data-set to the ``~/dataset`` folder.

    ```bash
    mkdir ~/dataset
    unzip sim-data.zip -d ~/dataset
    ```

4. Create traffic light data-set.

    ```bash
    cd CarND-Capstone/ros/src/tl_detector/light_classification
    python keras_yolo3/yolo_dataset.py --input ~/dataset --output ~/ --class_name "traffic light" --width 120 --height 240 --score 0.6 --model_path model_data/tiny_yolo.h5 --anchors_path keras_yolo3/model_data/tiny_yolo_anchors.txt
    ```

5. Train classifier using ``ipython``.

    ```ipython
    from tl_classifier import TLClassifier as TL
    tl=TL()
    tl.load_dataset('~/traffic_light_dataset')
    tl.train(filename='model_data/lenet_traffic_light.h5')
    ```

## Conclusion

This is very challenging project if you do not have an access to latest and greatest hardware to run it. Anyway I find it useful because it gives you the just a flavour of issues you may encounter in real live projects.
