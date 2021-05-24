# Build model deep learning using train dataset 
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Activation, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def restnet(n_class , input_shape = (224, 224, 3)):

    base_model=ResNet50(weights='imagenet', input_shape = input_shape,include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

    for layer in base_model.layers:
        layer.trainable = False

    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(512,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(128,activation='relu')(x) #dense layer 2
    preds=Dense(n_class,activation='softmax')(x) #final layer with softmax activation

    model=Model(inputs=base_model.input,outputs=preds)

    model.summary()

    return model

def vgg16(n_class , input_shape = (224, 224, 3)):

    base_model=VGG16(weights='imagenet', input_shape = input_shape,include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

    for layer in base_model.layers:
        layer.trainable = False

    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(512,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(128,activation='relu')(x) #dense layer 2
    preds=Dense(n_class,activation='softmax')(x) #final layer with softmax activation

    model=Model(inputs=base_model.input,outputs=preds)

    model.summary()

    return model

def mobilenet(n_class , input_shape = (224, 224, 3)):

    base_model=MobileNetV2(weights='imagenet', input_shape = input_shape,include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

    for layer in base_model.layers:
        layer.trainable = True

    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(512,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(128,activation='relu')(x) #dense layer 2
    preds=Dense(n_class,activation='softmax')(x) #final layer with softmax activation

    model=Model(inputs=base_model.input,outputs=preds)

    model.summary()

    return model