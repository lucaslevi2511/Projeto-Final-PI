import numpy as np
import cv2

def white_balance_gray_world(img, percent=1):
    result = img.copy().astype(np.float32)
    for c in range(3):
        low = np.percentile(result[:,:,c], percent)
        high = np.percentile(result[:,:,c], 100-percent)
        result[:,:,c] = np.clip(result[:,:,c], low, high)
        result[:,:,c] = (result[:,:,c] - low) * 255.0 / (high - low)
    return result.astype(np.uint8)

def gaussian(x_square, sigma):
    return np.exp(-0.5*x_square/sigma**2)

def bilateral_filter(image, sigma_space, sigma_intensity):
    kernel_size = int(2*sigma_space+1)
    half_kernel_size = int(kernel_size / 2)
    result = np.zeros(image.shape)
    W = 0

    for x in range(-half_kernel_size, half_kernel_size+1):
        for y in range(-half_kernel_size, half_kernel_size+1):
            Gspace = gaussian(x ** 2 + y ** 2, sigma_space)
            shifted_image = np.roll(image, [x, y], [1, 0])
            intensity_difference_image = image - shifted_image
            Gintenisity = gaussian(intensity_difference_image ** 2, sigma_intensity)
            result += Gspace*Gintenisity*shifted_image
            W += Gspace*Gintenisity
    return result / W

def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], 
                      [-.1687, -.3313, .5], 
                      [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def ycbcr2rgb(ycbcr_array):
    ycbcr = ycbcr_array.astype(np.float64)
    Y = ycbcr[:, :, 0]
    Cb = ycbcr[:, :, 1] - 128
    Cr = ycbcr[:, :, 2] - 128
    rgb = np.zeros_like(ycbcr)
    rgb[:, :, 0] = Y + 1.402 * Cr                
    rgb[:, :, 1] = Y - 0.344136 * Cb - 0.714136 * Cr 
    rgb[:, :, 2] = Y + 1.772 * Cb                 
    np.clip(rgb, 0, 255, out=rgb)
    return rgb.astype(np.uint8)

def calculate_histogram(image):
    histogram = np.zeros(256)
    for pixel_value in range(256):
        mask = (image == pixel_value)
        histogram[pixel_value] = np.sum(mask)
    return histogram

def normalizar(img):
    return img.astype(np.float32) / 255.0

def normalize_histogram(histogram, total_pixels):
    return histogram / total_pixels

def cumulative_distribution_function(normalized_histogram):
    return np.cumsum(normalized_histogram)

def histogram_equalization(image, cdf):
    return np.interp(image, np.arange(256), cdf * 255).astype(np.uint8)