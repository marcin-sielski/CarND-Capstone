from styx_msgs.msg import TrafficLight
import os
from datetime import datetime as dt
import h5py
from keras.models import load_model
import keras
from normalization import Normalization
from keras_yolo3.yolo import YOLO
import numpy as np
import tensorflow as tf

class TLClassifier(object):

    def __init__(self):
        #TODO load classifier
        model_data_path = os.path.dirname(os.path.abspath(__file__))
        self.detector = YOLO(anchors_path=model_data_path + 
        '/keras_yolo3/model_data/tiny_yolo_anchors.txt',
        model_path=model_data_path+'/model_data/tiny_yolo.h5',
        class_name='traffic light', height=240, width=120)
        model_name = model_data_path+'/model_data/lenet_traffic_light.h5'
        f = h5py.File(model_name, mode='r')
        model_version = f.attrs.get('keras_version')
        keras_version = str(keras.__version__).encode('utf8')

        if model_version != keras_version:
            print('You are using Keras version ', keras_version,
                ', but the model was built using ', model_version)

        self.classifier = load_model(model_name,
        custom_objects={'Normalization': Normalization()})
        global graph
        graph = tf.get_default_graph() 

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        _, images = self.detector.detect_image(image)
        with graph.as_default():
            if len(images) > 0:
                print(images[0].size)
                result = self.classifier.predict(np.asarray(images[0])[None, :, :, :],
                batch_size=1)
                ret=np.argmax(result)
                print(ret)
                return ret
        return TrafficLight.UNKNOWN

    def load_dataset(self, path):
        from sklearn.model_selection import train_test_split
        samples = []
        for root, _, files in os.walk(os.path.expanduser(path)):
            label = os.path.basename(root).lower()
            for file in files:
                sample = []
                sample.append(root+'/'+file)
                sample.append(label)
                samples.append(sample)
        self.train_samples, self.validation_samples = train_test_split(samples,
        test_size=0.2)
        train = len(self.train_samples)
        validation = len(self.validation_samples)
        total = train + validation
        print('Train set size: {} ({}%)'.format(train, round(train*100/total)))
        print('Validation set size: {} ({}%)'.format(validation,
        round(validation*100/total)))
        print('Total size: {} (100%)'.format(total))

    def generator(self, samples, batch_size=16):
        label_dict = {
            "none": TrafficLight.UNKNOWN,
            "green": TrafficLight.GREEN,
            "red": TrafficLight.RED,
            "yellow": TrafficLight.YELLOW
        }
        from random import shuffle
        import cv2
        import sklearn
        import numpy as np
        from keras.utils import to_categorical
        num_samples = len(samples)
        while 1:
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]
                images=[]
                labels=[]
                for batch_sample in batch_samples:
                    image = cv2.imread(batch_sample[0])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    label = batch_sample[1]
                    images.append(image)
                    labels.append(to_categorical(label_dict[label], num_classes=4))

                X_train = np.array(images)
                y_train = np.array(labels)
                yield sklearn.utils.shuffle(X_train, y_train)

    def train(self, filename=os.path.dirname(os.path.abspath(__file__))+
    '/model_data/lenet_'+str(dt.now())+'.h5', batch_size=16,
    epochs=15):
        from keras.models import Sequential
        from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, \
        Cropping2D, Dropout
        from keras.callbacks import ModelCheckpoint
        from keras.utils import plot_model
        from math import ceil
        model = Sequential()
        model.add(Normalization(input_shape=(240, 120, 3))) 
        model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu'))
        model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
        model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dropout(0.5))
        model.add(Dense(50))
        model.add(Dropout(0.5))
        model.add(Dense(4))
        model.compile(loss='mse', optimizer='adam',
        metrics=['accuracy'])
        history = model.fit_generator(self.generator(self.train_samples,
        batch_size=batch_size), 
        steps_per_epoch=ceil(len(self.train_samples)/batch_size),
        validation_data=self.generator(self.validation_samples, 
        batch_size=batch_size),
        validation_steps=ceil(len(self.validation_samples)/batch_size),
        epochs=epochs, verbose=1, callbacks=[ModelCheckpoint(filename, verbose=1,
        save_best_only=True)])
        print(history.history.keys())
        print('Loss:')
        print(history.history['loss'])
        print('Validation Loss:')
        print(history.history['val_loss'])
        print('Accuracy:')
        print(history.history['acc'])
