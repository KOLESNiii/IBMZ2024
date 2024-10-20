import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Define color thresholds
def classify_image(image):
    if image is None:
        print("Error: Image not loaded properly.")
        return 'unknown'

    # Convert image to HSV color space for better color recognition
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Adjusted HSV color ranges for different environments
    green_range = ((35, 20, 20), (85, 255, 255))  # Rainforest (green, with lower brightness threshold)
    yellow_range = ((15, 40, 40), (35, 255, 255))  # Desert (yellow-tan, more sensitive)
    blue_range = ((85, 30, 30), (135, 255, 255))  # Ocean (expanded blue range)

    # Function to calculate percentage of pixels in a given color range
    def calculate_color_percentage(hsv_img, lower_bound, upper_bound):
        mask = cv2.inRange(hsv_img, np.array(lower_bound), np.array(upper_bound))
        color_pixels = cv2.countNonZero(mask)
        total_pixels = hsv_img.shape[0] * hsv_img.shape[1]
        return (color_pixels / total_pixels) * 100

    # Calculate the percentage of each color
    green_percentage = calculate_color_percentage(hsv_image, *green_range)
    yellow_percentage = calculate_color_percentage(hsv_image, *yellow_range)
    blue_percentage = calculate_color_percentage(hsv_image, *blue_range)

    # Texture Analysis using Local Binary Pattern (LBP)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))

    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    # Threshold for texture classification (adjust based on texture data)
    if hist[0] > 0.5:  # Very smooth texture likely indicates desert
        texture_classification = 'desert'
    else:
        texture_classification = 'unknown'

    # Print out percentages for debugging
    print(f"Green: {green_percentage:.2f}%, Yellow: {yellow_percentage:.2f}%, Blue: {blue_percentage:.2f}%")
    print(f"Texture Classification: {texture_classification}")

    # Combine color and texture classification
    if green_percentage > 20:
        return 'rainforest'
    elif yellow_percentage > 20:
        return 'desert'
    elif blue_percentage > 20:
        return 'ocean'
    else:
        return texture_classification  # Fallback to texture classification if color is unclear

# Path to the images folder
images_folder = "/home/ashwin/ibmz/IBMZ2024/images"

# Iterate through the images in the folder
for image_file in os.listdir(images_folder):
    if image_file.endswith(('.png', '.jpg', '.jpeg')):  # Check if the file is an image
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)
        
        # Check if the image was loaded successfully
        if image is None:
            print(f"Failed to load image: {image_file}")
            continue
        
        # Classify the image
        classification = classify_image(image)
        
        # Output the result
        print(f"Image: {image_file} -> Category: {classification}")


# /home/ashwin/ibmz/IBMZ2024/images
