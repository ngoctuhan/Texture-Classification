import numpy as np, cv2 

def zigzag(I2):
    
    """
    Calculator zigzag for image 
    
    Parameter:
        - I2: image input with shape (width, height)

    Return:
        - Image after transform zigzag
    """
    
    m, n = I2.shape[0], I2.shape[1]
    I3 = np.ones_like(I2)
    result = np.zeros((m-1, n-1), dtype=int)
    
    for i in range(1,m-1):
        for j in range(1, n-1):
            J0 = I2[i, j]
            
            I3[i-1,j-1]=I2[i-1,j-1]>J0; 
            I3[i-1,j]=I2[i-1,j]>J0;

            I3[i-1,j+1]=I2[i-1,j+1]>J0; 
            I3[i,j+1]=I2[i,j+1]>J0;5

            I3[i+1,j+1]=I2[i+1,j+1]>J0; I3[i+1,j]=I2[i+1,j]>J0; I3[i+1,j-1]=I2[i+1,j-1]>J0; I3[i,j-1]=I2[i,j-1]>J0;

            result[i,j] = I3[i-1,j-1]*2**0 + I3[i-1,j]*2**1 + I3[i,j-1]*2**2 + I3[i+1,j-1]*2**3 + I3[i-1,j+1]*2**4 + I3[i,j+1]*2**5 + I3[i+1,j]*2**6 + I3[i+1,j+1]*2**7;
            
    return np.array(result)