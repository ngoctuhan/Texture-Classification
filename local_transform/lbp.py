from skimage.feature import local_binary_pattern

def lbp(image, radius = 3):


    """
    Transform image using LBP

    Parameters:
        - image: input image with shape (width, height)

    Return:
        - IMAGE after transform
    """
    n_points = 8 * radius


    lbp = local_binary_pattern(image, n_points, radius, 'uniform')

    return lbp 
