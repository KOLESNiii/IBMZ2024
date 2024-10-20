import cv2 as opencv
from sentinelhub import SHConfig, MimeType, CRS, BBox, SentinelHubRequest, DataCollection, bbox_to_dimensions
import numpy as np
import matplotlib.pyplot as plt
import rasterio

SEG_HYPER_PARAMS = {
    "kernel_size" : {"desert" : (10, 10), "rainforest" : ()},
    "threshold_factor" : {"desert" : 0.03, "rainforest" : None}
}


class imageBundle:
    def __init__(self, img, year, coords):
        self.img = img  # Assuming img is a grayscale image loaded with OpenCV
        self.year = year
        self.coords = coords

    def toNdvi(self):
        # Check if the image is valid
        if self.img is None:
            raise ValueError("No image data found.")

        # Normalize the grayscale image to a range [0, 1]
        gray_normalized = self.img.astype(float) / 255.0

        # Map normalized values to NDVI range [-1, 1]
        ndvi = gray_normalized * 2 - 1  # Scale from [0, 1] to [-1, 1]

        return ndvi  # Return NDVI values
    
    def returnNdvi(self):
        """Return a 1D array of NDVI values."""
        ndvi = self.toNdvi()  # Get the NDVI array
        ndvi_flat = ndvi.flatten()  # Flatten the NDVI array to 1D
        return ndvi_flat  # Return the flattened NDVI values
    
    def greyScaleSegment(self):
        """Performs image segmentation and returns monochrome heat map"""
        hsv = opencv.cvtColor(self.img, opencv.COLOR_BGR2HSV)
        
        # Define HSV range for green
        lower_green = np.array([10, 30, 30])
        upper_green = np.array([105, 255, 255])

        green_mask = opencv.inRange(hsv, lower_green, upper_green)
        
        green_area_count = np.count_nonzero(green_mask)
        
        # Apply the mask to the original image (focus only on green and blue regions)
        filtered_img = opencv.bitwise_and(self.img, self.img, mask=green_mask)
        
        min_green_area_threshold = 3700
        
        if green_area_count > min_green_area_threshold:
        #if True:
            gray = opencv.cvtColor(filtered_img, opencv.COLOR_BGR2GRAY)
            
            _, thresh = opencv.threshold(gray, 0, 255, opencv.THRESH_BINARY_INV + opencv.THRESH_OTSU)

            # noise removal
            kernel = np.ones((10, 10), np.uint8)
            opening = opencv.morphologyEx(thresh, opencv.MORPH_OPEN, kernel, iterations = 2)
            
            # sure background area
            sure_bg = opencv.dilate(opening, kernel, iterations=3)
            
            # Finding sure foreground area
            dist_transform = opencv.distanceTransform(opening, opencv.DIST_L2, 5)
            _, sure_fg = opencv.threshold(dist_transform, 0.03 * dist_transform.max(), 255, 0)
            
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = opencv.subtract(sure_bg, sure_fg)

            # Marker labelling
            _, markers = opencv.connectedComponents(sure_fg)
            
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1
            
            # Now, mark the region of unknown with zero
            markers[unknown == 255] = 0

            markers = opencv.watershed(self.img, markers)
            self.img[markers == -1] = [255, 0, 0]

            mask = np.zeros_like(gray)
            mask[markers > 1] = 255
            
            return opencv.bitwise_not(mask)
        else:
            return np.zeros_like(self.img)

    def display(self):
        """Method to display the original image and NDVI."""
        ndvi_image = self.toNdvi()

        # Normalize NDVI for display purposes (to [0, 255])
        ndvi_display = opencv.normalize(ndvi_image, None, 0, 255, opencv.NORM_MINMAX).astype(np.uint8)

        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(self.img, cmap='gray')
        plt.title(f'Original Grayscale Image ({self.year})')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(ndvi_display, cmap='hot')
        plt.title('Mapped NDVI')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(self.greyScaleSegment(), cmap='hot')
        plt.title('Monochrome segmentation')
        plt.axis('off')

        plt.colorbar()
        plt.show()

if __name__ == "__main__":
    img = opencv.imread('actualImages/2024img6808.jpeg')
    #img = opencv.imread('images/testImage4-sahel.jpeg')
    imgObj = imageBundle(img, 2024, (0, 0))
    
    #'''
    plt.subplot(1, 2, 1)
    #plt.imshow(opencv.split(img)[1])
    plt.imshow(img)
    
    plt.subplot(1, 2, 2)
    plt.imshow(imgObj.greyScaleSegment())
    
    plt.show()
    
    plt.imshow(imgObj.greyScaleSegment())
    plt.show()
    #'''
    
    #imgObj.greyScaleSegment()