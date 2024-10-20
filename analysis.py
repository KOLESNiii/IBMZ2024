from imageBundle import imageBundle, SEG_HYPER_PARAMS
import numpy as np
import cv2 as opencv
import matplotlib.pyplot as plt

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

    def demosaic(img):
        sobel_x = opencv.Sobel(img, opencv.CV_64F, 1, 0, ksize=19)
        seam_strengths = np.sum(np.abs(sobel_x), axis=0)
        seam_strengths_smoothed = opencv.GaussianBlur(seam_strengths, (51, 51), 0)
        seam_strengths_smoothed = seam_strengths_smoothed
        
        max_threshold = np.mean(seam_strengths_smoothed) + 3 * np.std(seam_strengths_smoothed)
        min_threshold = np.mean(seam_strengths_smoothed) - 1.9 * np.std(seam_strengths_smoothed)
        #threshold = np.quantile(seam_strengths_smoothed, 0.75)
        #print(sobel_x)
        #potential_seams = np.where((seam_strengths_smoothed > max_threshold) | (seam_strengths_smoothed < min_threshold))[0]
        potential_seams = np.where(seam_strengths_smoothed > max_threshold)[0]
        
        if len(potential_seams) > 0:
            seam_x = potential_seams[np.argmax(seam_strengths_smoothed[potential_seams])]
            
            image_with_seam = opencv.cvtColor(img, opencv.COLOR_GRAY2BGR)
            opencv.line(image_with_seam, (seam_x, 0), (seam_x, img.shape[0]), (0, 0, 255), 2)

            plt.imshow(opencv.cvtColor(image_with_seam, opencv.COLOR_BGR2RGB))
            plt.title(f"Detected Seam at x = {seam_x}")
            print(f"Detected Seam at x = {seam_x}")
            plt.show()
            
            plt.plot(seam_strengths_smoothed)
            plt.title("Seam Strengths along x-axis (Sobel)")
            plt.xlabel("x-coordinate")
            plt.ylabel("Sum of Sobel Gradients")
            plt.show()
        
        else:
            print("No significant seam detected")
        
        
        
def compare(imgBundle1, imgBundle2):
    return [SimpleAnalysisTools.getWhitePercentage(imgBundle2.greyScaleSegment()) 
            - SimpleAnalysisTools.getWhitePercentage(imgBundle1.greyScaleSegment()), 
      SimpleAnalysisTools.getMeanNDVIDiff(imgBundle1.returnNdvi(), imgBundle2.returnNdvi())]

if __name__ == "__main__":
    img = opencv.imread('images/testImage4-sahel.jpeg')
    img2 = opencv.imread('images/testImage5-us.jpeg')
    imgObj = imageBundle(img, 2024, (0, 0))
    imgObj2 = imageBundle(img2, 2024, (0, 0))
    
    SimpleAnalysisTools.demosaic(opencv.imread('images/testImage1-amazon.jpeg', opencv.IMREAD_GRAYSCALE))
    SimpleAnalysisTools.demosaic(opencv.imread('images/testImage3-sahel.jpeg', opencv.IMREAD_GRAYSCALE))
    SimpleAnalysisTools.demosaic(opencv.imread('images/testImage2-amazon.jpeg', opencv.IMREAD_GRAYSCALE))
    SimpleAnalysisTools.demosaic(opencv.imread('images/testImage5-us.jpeg', opencv.IMREAD_GRAYSCALE))
    SimpleAnalysisTools.demosaic(opencv.imread('images/ndvi1.jpeg', opencv.IMREAD_GRAYSCALE))
    SimpleAnalysisTools.demosaic(opencv.imread('images/testImage6-europe.jpeg', opencv.IMREAD_GRAYSCALE))
    
    #plt.imshow(opencv.imread('images/colour1.jpeg', opencv.IMREAD_REDUCED_GRAYSCALE_2))
    #plt.show()
    
    '''
    comparison = compare(imgObj, imgObj2)
    print(f"US forest coverage - Sahel forest coverage: {comparison[0]}")
    print(f"NDVI diff: {comparison[1]}")
    imgObj2.display()
    
    opencv.imshow("Greyscale segmentation", imgObj.greyScaleSegment())
    opencv.waitKey(0)
    opencv.destroyAllWindows()
    '''
