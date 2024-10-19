from imageBundle import imageBundle, SEG_HYPER_PARAMS
import numpy as np
import cv2 as opencv

class SimpleAnalysisTools:
    def getWhitePercentage(img):
        # Count the number of white pixels (255)
        white_pixels = np.sum(img == 255)
        
        # Calculate the total number of pixels in the image
        total_pixels = img.shape[0] * img.shape[1]
        
        # Calculate the percentage of white pixels
        white_percentage = (white_pixels / total_pixels) * 100
        
        return white_percentage
    
    def getMeanNDVIDiff(ndvi1, ndvi2):
        return np.mean(ndvi2) - np.mean(ndvi1)

def compare(imgBundle1, imgBundle2):
    return [SimpleAnalysisTools.getWhitePercentage(imgBundle2.greyScaleSegment()) 
            - SimpleAnalysisTools.getWhitePercentage(imgBundle1.greyScaleSegment()), 
      SimpleAnalysisTools.getMeanNDVIDiff(imgBundle1.returnNdvi(), imgBundle2.returnNdvi())]

if __name__ == "__main__":
    img = opencv.imread('images/testImage4-sahel.jpeg')
    img2 = opencv.imread('images/testImage5-us.jpeg')
    imgObj = imageBundle(img, 2024, (0, 0))
    imgObj2 = imageBundle(img2, 2024, (0, 0))
    
    comparison = compare(imgObj, imgObj2)
    print(f"US forest coverage - Sahel forest coverage: {comparison[0]}")
    print(f"NDVI diff: {comparison[1]}")
    imgObj2.display()
    '''
    opencv.imshow("Greyscale segmentation", imgObj.greyScaleSegment())
    opencv.waitKey(0)
    opencv.destroyAllWindows()
    '''
