import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
import numpy

def one_hot_label(y_root, y_follow):

    """
    Covert label of list data to one hot vector 
    Parameters:
        - y_root: label using root
        - y_follow: label follow root
    """

    gle = LabelEncoder()

    labels_root = gle.fit_transform(y_root)
    labels_follow = gle.transform(y_follow)

    mappings = { index: label for index, label in enumerate(gle.classes_)}
    print(mappings)

    label_binary = LabelBinarizer()

    y_train = label_binary.fit_transform(labels_root)
    y_test = label_binary.transform(labels_follow)
    
    return y_train, y_test 

def draw_distribute_data(xAxis, yAxis, name_save=  None):
    
    """
    Draw bar chart to visualize distribute of dataset 
    Parameters:
        - xAxis: dataset of x axis usually: name label, year, num_epoch, ...
        - yAxis: frequently of objects
    """

    plt.bar(xAxis,yAxis)
    plt.title('Histogram')
    
    plt.xlabel('Value')
    plt.ylabel('frequently')
    
    plt.show()

    if name_save is not None:
        plt.savefig(name_save)

def historgram(image):

    """
    Calculator histogram of gray image 
    Paramters:
        - image: input image with shape = (width, height)

    Return:
        - Output: a vector 256 elements 
    """

    unique, counts = numpy.unique(image, return_counts=True)
    print(dict(zip(unique, counts)))
    return counts