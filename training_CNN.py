from model_cnn.models import *
from model_cnn.metrics import *
from model_cnn.loss import * 
from utils.loader import load_for_train 
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Choose pretrained model')
    parser.add_argument('--pretrain_model', type=str,
                        help='pre-trained model vgg16 restnet50 mobilenet')
    parser.add_argument('--dataset_name', type=str,
                        help='name of dataset')
    parser.add_argument('--type_output', type=str,
                        help='type of dataset for train zigzag or raw')
    parser.add_argument('--type', type=str,
                        help='type of data raw or zigzag')

    args = parser.parse_args()


    X_train, y_train, X_test, y_test = load_for_train('dataset/train', 'dataset/valid', args.type_output, args.dataset_name)
    n_class =  y_train.shape[1]

    if args.pretrain_model == 'vgg16':
        model = vgg16(y_train.shape[1], X_train.shape[:-1])
    elif args.pretrain_model =='restnet':
        model = restnet(y_train.shape[1], X_train.shape[:-1])
    else:
        model = mobilenet(y_train.shape[1], X_train.shape[:-1])

    model.compile(loss=[categorical_focal_loss(alpha=[[.25] * n_class], gamma=2)], metrics=["accuracy", f1_m], optimizer='adam')
    # model.compile(loss='categorical_crossentropy', metrics=["accuracy", f1_m], optimizer='adam')
    history = model.fit(X_train, y_train, batch_size=16, epochs=50, validation_data=(X_test, y_test))

    loss_train = history.history['train_loss']
    loss_val = history.history['val_loss']
    epochs = range(1,35)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



