import cv2
import matplotlib.pyplot as plt


def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_image_plt(title, image, cmap = None):
    plt.figure(title)
    plt.imshow(image,cmap=cmap)
    plt.axis('off')
    plt.show()

def cvt_image_colorspace(image, colorspace = cv2.COLOR_BGR2GRAY):
    return cv2.cvtColor(image, colorspace)

def median_filtering(image, kernel_size=3):
    '''

    :param image: grayscale image
    :param kernel_size: kernel size should be odd number
    :return: blurred image
    '''

    return cv2.medianBlur(image, kernel_size)


def apply_threshold(image, **kwargs):
    '''

    :param image: image object
    :param kwargs: threshold parameters - dictionary
    :return:
    '''
    threshold_method = kwargs['threshold_method']
    max_value = kwargs['pixel_value']
    threshold_flag = kwargs.get('threshold_flag', None)
    if threshold_flag is not None:
        ret, thresh1 = cv2.adaptiveThreshold(image, max_value, threshold_method,cv2.THRESH_BINARY, kwargs['block_size'], kwargs['const'])
    else:
        ret, thresh1 = cv2.threshold(image, kwargs['threshold'], max_value, threshold_method)
    return thresh1

def sobel_filter(img,x,y,kernel_size = 3):
    return cv2.Sobel(img, cv2.CV_8U, x,y, ksize=kernel_size)