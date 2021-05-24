import os, cv2, numpy as np 
import threading 
from utils.utils import historgram
from local_transform.zigzag import zigzag
from local_transform.lbp import lbp
from utils.utils import one_hot_label

def load_data(train_path, val_path, type_output = 'hist', name_dataset = 'KTH', return_data = 'False'):

    """
    Load and processing dataset and save in run to can training model classification

    Paramters:
        - train_path: path of data for train
        - val_path : path of data for valuation
        - type_output: type of data after load. Can choose history or zigzag, raw corresponding is: hist, zigzag, raw
        - name_dataset: have 3 datasets, choose name of dataset need load, can load all if using name_dataset = ''
    
    Return:
        X_train, y_train, X_test, y_test
    """

    X_train, y_train, X_test, y_test = [], [] ,  [], []

    def load(path): 
        
        X, y= [], []
        for folder in os.listdir(path):

            if folder.split('_')[0] != name_dataset:
                continue

            if name_dataset == 'UIUC':
                if folder.split('_')[0] == 'KTH' or folder.split('_')[0] == 'Kyberge':
                    continue

            path_folder = os.path.join(path, folder)
          
            print("Loading....", folder.split('_')[1])
            for filename in os.listdir(path_folder):

                file_path = os.path.join(path_folder, filename)
                img =  cv2.resize(cv2.imread(file_path), (256,256))

                if type_output == 'zigzag':
                    x = np.zeros_like(img)
                    for chanel in range(3):
                        tmp = img[:, :, chanel]
                        if tmp.shape != (256, 256):
                            return
                        x[:, :, chanel] = zigzag(tmp)
                    X.append(x)
                    y.append(folder)
                
                elif type_output == 'hist_zigzag':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    tmp = historgram(zigzag(img))
                    X.append(tmp)
                    y.append(folder)
                
                elif type_output == 'raw':
                    X.append(img)
                    y.append(folder)
                
                elif type_output == 'lbp':
                    x = np.zeros_like(img)
                    for chanel in range(3):
                        tmp = img[:, :, chanel]
                        if tmp.shape != (256, 256):
                            return
                        x[:, :, chanel] = lbp(tmp)
                    X.append(x)

                elif type_output == 'hist_lbp':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    tmp = historgram(lbp(img))
                    X.append(tmp)
                    y.append(folder)
                else:
                    ft = []
                    for chanel in range(3):
                        tmp = img[:, :, chanel]
                        ft += historgram(tmp)
    
                    X.append(ft)
                    y.append(folder)

                if len(X) % 100 == 0:
                    print(">", end="")
            print()
        
        return np.array(X), np.array(y)

    #load data for training
    def thread1():
        
        #load data for train
        X_train, y_train = load(train_path)
        
        np.save('dataset/train_{}_{}.npy'.format(name_dataset, type_output), X_train)
        np.save('dataset/lbl_train_{}_{}.npy'.format(name_dataset, type_output), y_train)


    def thread2():
        
        #load data for val
        X_test, y_test = load(val_path)

        np.save('dataset/test_{}_{}.npy'.format(name_dataset, type_output), X_test)
        np.save('dataset/lbl_test_{}_{}.npy'.format(name_dataset, type_output), y_test)
    
    x1= threading.Thread(target=thread1)
    x2 = threading.Thread(target=thread2)

    x1.start()
    x2.start()

    if return_data == True:
        return X_train, y_train, X_test, y_test
    
    else:
        return None
    
# load_data('dataset/train', 'dataset/valid')

def load_for_train(train_path, val_path, type_output = 'hist', name_dataset = 'KTH', type = 'deep'):

    """
    Load data from disk to RAM 

    Paramters:
        - train_path: path of data train
        - val_path  : path of data valuation
        - type: type of label for model deep or model machine learning
    """
    X_train, y_train, X_test, y_test = load_data(train_path, val_path, type_output, name_dataset)

    if type == 'deep':
        y_train, y_test = one_hot_label(y_train, y_test)

    return X_train, y_train, X_test, y_test