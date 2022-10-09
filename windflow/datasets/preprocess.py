import numpy as np

def image_histogram_equalization(image, number_bins=500):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    image_equalized[~np.isfinite(image_equalized)] = 0. 
    
    return image_equalized.reshape(image.shape)#, cdf

def normalize_qv(qv):
    qv_log = np.log(qv * 1e4 + 0.001)
    qv_log_norm = (qv_log - 1.06) / 2.15
    return qv_log_norm