import cv2 as opencv
from sentinelhub import SHConfig, MimeType, CRS, BBox, SentinelHubRequest, DataCollection, bbox_to_dimensions
import numpy as np
import matplotlib.pyplot as plt
import rasterio


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

    def display(self):
        """Method to display the original image and NDVI."""
        ndvi_image = self.toNdvi()

        # Normalize NDVI for display purposes (to [0, 255])
        ndvi_display = opencv.normalize(ndvi_image, None, 0, 255, opencv.NORM_MINMAX).astype(np.uint8)

        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(self.img, cmap='gray')
        plt.title(f'Original Grayscale Image ({self.year})')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(ndvi_display, cmap='hot')
        plt.title('Mapped NDVI')
        plt.axis('off')

        plt.colorbar()
        plt.show()