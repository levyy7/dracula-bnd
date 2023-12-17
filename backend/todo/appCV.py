import cv2 as cv
import numpy as np 
from io import BytesIO
from PIL import Image
import math


# Returns: 0-25->"LOw", 25-75->"MEDIUM", 75-100->"HIGH",
def computeBloodAmount(decoded_data):
    # Create a BytesIO object
    image_stream = BytesIO(decoded_data)

    # Open the image using PIL (Python Imaging Library)
    image = Image.open(image_stream)

    # Convert the PIL image to a NumPy array
    im = np.array(image)

    height, width, channels = im.shape
    factor = math.sqrt((400*400)/(height*width))
    im = cv.resize(im, None, fx=factor, fy=factor)
    #im = cv .resize(im, (400, 400))
    
    segmented_image = apply_kmeans(im, 5)
    cv.imwrite("segmented_image.png", segmented_image)

    maskRed = top_red_pixels_mask(im, 500)
    bp = getBloodPixels(segmented_image, maskRed)

    maskWhite = top_white_pixels_mask(im, 500)
    wp = getBloodPixels(segmented_image, maskWhite)

    blood_ratio = (bp/(bp + wp))
    
    if blood_ratio < 0.25:
        return "LOW"
    elif blood_ratio < 0.75:
        return "MEDIUM"
    else:
        return "HIGH"


def apply_kmeans(image, K):
    Z = image.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    _,label,center = cv.kmeans(Z,K,None,criteria,15,cv.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))

    #labellist = label.flatten().tolist()

    #cv.imshow('res2',res2)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    return res2

# Returns a binary image equal size of the original that has the most red pixels to 1
def top_red_pixels_mask(image, top):
    # Convert the image from BGR to RGB
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Create a mask for red pixels
    lower_red = np.array([150,0,0])
    upper_red = np.array([255,100,100])
    red_mask = cv.inRange(image_rgb, lower_red, upper_red)


    # Find the coordinates of the red pixels
    red_pixel_coordinates = np.column_stack(np.where(red_mask > 0))

    # Sort the red pixels by their intensity (sum of RGB values)
    sorted_red_pixels = sorted(red_pixel_coordinates, key=lambda x: np.sum(image_rgb[x[0], x[1]]), reverse=True)

    # Get the top 10 red pixels
    pixels = sorted_red_pixels[:top]

    im = image.copy()
    im = cv.bitwise_xor(im, im)
    for position in pixels:
        im[position[0], position[1]] = 255

    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    #im = np.array(im)
    #cv.imshow('res2',im)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    return im


# Returns a binary image equal size of the original that has the most red pixels to 1
def top_white_pixels_mask(image, top):
    # Convert the image from BGR to RGB
    im = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Flatten the 2D array of pixel intensities to a 1D array
    pixel_values = im.flatten()

# Get the indices of the sorted pixel intensities in descending order
    sorted_indices = np.argsort(pixel_values)[::-1]

    # Extract the coordinates of the 10 whitest pixels

    pixels = [np.unravel_index(index, im.shape) for index in sorted_indices[:top]]

    im = image.copy()
    im = cv.bitwise_xor(im, im)
    for position in pixels:
        im[position[0], position[1]] = 255

    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    #im = np.array(im)
    #cv.imshow('res2',im)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    return im


def getBloodPixels(image, mask):
    # Get list of unique colours
    image_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    uniquecols = np.unique(image_bgr.reshape(-1,3), axis=0) 

    maxPixels = 0
    # Iterate over unique colours
    for i, c in enumerate(uniquecols):
        # Make output image white wherever it matches this colour, and black elsewhere
        result = np.where(np.all(image_bgr==c,axis=2)[...,None], 255, 0)
        result = np.uint8(result)

        #cv.imshow('res2',result)
        #cv.waitKey(0)
        #cv.destroyAllWindows()

        reconstruction = morphological_reconstruction(result, mask)

        #cv.imshow('res2',reconstruction)
        #cv.waitKey(0)
        #cv.destroyAllWindows()

        maxPixels = maxPixels + reconstruction.flatten().tolist().count(255)

    return maxPixels


def morphological_reconstruction(image, marker):
    # Ensure the input images have the same size and type
    marker = np.uint8(marker)
    mask = np.uint8(image)

    # Perform morphological reconstruction
    reconstruction = cv.bitwise_and(mask, marker)

    # Create a structuring element for the morphological operations
    kernel = np.ones((3, 3), np.uint8)

    # Perform dilation on the marker image
    marker = cv.dilate(marker, kernel, iterations=1)

    # Iterate until there is no change in the reconstruction
    while not np.array_equal(reconstruction, marker):
        marker = reconstruction.copy()
        reconstruction = cv.bitwise_and(mask, cv.dilate(marker, kernel, iterations=1))

    return reconstruction


#np.set_printoptions(threshold=np.inf)
#im = cv.imread("test_images/test6_cropped.jpg")
#print(computeBloodAmount("test_images/test7_cropped.jpg"))

#for i in range(1, 10):
#    filename = f"test_images/test{i}_cropped.jpg"
#    print(f"Computing amount of blood in image {filename}")

#    print(computeBloodAmount(filename))


# Get 10 points where red is stronger and then reconstruct using labels 