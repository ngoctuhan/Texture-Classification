from utils.loader import load_for_train 
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
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


    print ('Training data shape: ', X_train.shape)
    print ('Training labels shape: ', y_train.shape)
    print ('Test data shape: ', X_test.shape)
    print ('Test labels shape: ', y_test.shape)

  
    clf = SVC(gamma='auto').fit(X_train, y_train)
    # clf = LinearSVC(C = 200).fit(X_train, y_train)
    pre = clf.predict(X_test)
    print ("Accuracy for validation : %.2f %%" %(100*accuracy_score(y_test, pre.tolist())) ) 
