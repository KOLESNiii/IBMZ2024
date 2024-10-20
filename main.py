import cv2 as opencv
import os
import math
import geopandas as gpd
from shapely.geometry import box, shape
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage.feature import local_binary_pattern
import re

BATCH_SIZE = 32

def imageToBundle(image, ndviImage):
    img = opencv.imread(image)
    ndviImg = opencv.imread(ndviImage)
    if image.startswith("NDVI"):
        year = int(image[4:8])
    else:
        year = int(image[0:4])
    pattern = r"img(\d+)\.jpeg"
    coords = get_nth_grid(re.search(pattern, image).group(1),225)
    return imageBundle(img, ndviImg, coords, year)

def batch_image_loader(folder):
    # Regex patterns to match NDVI and non-NDVI files for 2016 and 2024
    ndvi_pattern = r"NDVI(2016|2024)img(\d+)\.jpeg"
    non_ndvi_pattern = r"(2016|2024)img(\d+)\.jpeg"

    files = os.listdir(folder)
    
    # Dictionaries to store files by (year, img number)
    ndvi_files = {}
    non_ndvi_files = {}
    
    # Populate dictionaries
    for file in files:
        ndvi_match = re.match(ndvi_pattern, file)
        non_ndvi_match = re.match(non_ndvi_pattern, file)
        
        if ndvi_match:
            year, img_num = ndvi_match.groups()
            ndvi_files[(year, img_num)] = file
        elif non_ndvi_match:
            year, img_num = non_ndvi_match.groups()
            non_ndvi_files[(year, img_num)] = file

    # Generator to yield batches of size 32 or less
    def batch_generator():
        # Iterate over the unique image numbers in the files
        for img_num in set(img_num for year, img_num in ndvi_files.keys()):
            # Check if all required files for 2016 and 2024 exist for this img_num
            if (('2016', img_num) in ndvi_files and ('2016', img_num) in non_ndvi_files and 
                ('2024', img_num) in ndvi_files and ('2024', img_num) in non_ndvi_files):
                
                ndvi_file_2016 = os.path.join(folder, "NDVI"+ndvi_files[('2016', img_num)])
                non_ndvi_file_2016 = os.path.join(folder, non_ndvi_files[('2016', img_num)])
                ndvi_file_2024 = os.path.join(folder, "NDVI"+ndvi_files[('2024', img_num)])
                non_ndvi_file_2024 = os.path.join(folder, non_ndvi_files[('2024', img_num)])
                
                yield (ndvi_file_2016, non_ndvi_file_2016, ndvi_file_2024, non_ndvi_file_2024)
    
    # Creating and returning batches of size 32 from the generator
    batch = []
    for files_group in batch_generator():
        batch.append(files_group)
        if len(batch) == 32:
            yield batch
            batch = []
    
    # Yield the remaining batch if any left after the loop
    if batch:
        yield batch

def load_images(batch):
    images = []
    for ndvi_2016, non_ndvi_2016, ndvi_2024, non_ndvi_2024 in batch:
        try:
            image_2016 = imageToBundle(non_ndvi_2016, ndvi_2016)
            image_2024 = imageToBundle(non_ndvi_2024, ndvi_2024)
            images.append((image_2016, image_2024))
        except Exception as e:
            print(f"Error loading images {ndvi_2016}, {non_ndvi_2016}, {ndvi_2024}, or {non_ndvi_2024}: {e}")
    return images


    # Process the images as needed
# Constants
earth_radius = 6371.0  # Earth radius in km
sizeInKM = 225
km_per_deg_lat = 111.194927  # Kilometers per degree of latitude
goejson_data = gpd.read_file('landSimplified.json')
# Function to calculate the longitude step at a given latitude
def calculate_longitude_step(lat, side_km):
    km_per_deg_lon = km_per_deg_lat * math.cos(math.radians(lat))  # Kilometers per degree of longitude at latitude
    return side_km / km_per_deg_lon


# Generate grid from bottom-left (-90, -180) to top-right (90, 180)
def get_grids(side_length_km = 85, first=0, last=-1):
    # Latitude step (constant since km per degree latitude is constant)
    lat_step = side_length_km / km_per_deg_lat
    lat = -60
    count = 0
    while lat + lat_step <= 70:
        lon = -180
        while lon + calculate_longitude_step(lat, side_length_km) <= 180:
            if last != -1 and count > last:
                return
            lon_step = calculate_longitude_step(lat, side_length_km)
            
            # Bottom-left and top-right corners of the box
            bottom_left = [lon, lat]
            top_right = [lon + lon_step, lat + lat_step]
            
            if count >= first:
                if goejson_data.intersects(box(bottom_left[0], bottom_left[1], top_right[0], top_right[1])).any():
                    # return a generator
                    yield (count, bottom_left + top_right)
            
            # Move to the next box in longitude
            lon += lon_step
            count += 1
        # Move to the next box in latitude
        lat += lat_step
    

def get_nth_grid(n, side_length_km = 85):
    # Latitude step (constant since km per degree latitude is constant)
    lat_step = side_length_km / km_per_deg_lat
    lat = -60
    count = 0
    while lat + lat_step <= 70:
        lon = -180
        while lon + calculate_longitude_step(lat, side_length_km) <= 180:
            lon_step = calculate_longitude_step(lat, side_length_km)
            if count == n:
                return [lon, lat, lon + lon_step , lat + lat_step]


            lon += lon_step
            count += 1
        # Move to the next box in latitude
        lat += lat_step
SEG_HYPER_PARAMS = {
    "kernel_size" : {"desert" : (10, 10), "rainforest" : ()},
    "threshold_factor" : {"desert" : 0.03, "rainforest" : None}
}


class imageBundle:
    def __init__(self, img, ndviImg, year, coords):
        self.img = img
        self.ndviImg = ndviImg  # Assuming img is a grayscale image loaded with OpenCV
        self.year = year
        self.coords = coords
        #self.classification = self.classify_image()

    def toNdvi(self):
        # Check if the image is valid
        if self.ndviImg is None:
            raise ValueError("No image data found.")

        # Normalize the grayscale image to a range [0, 1]
        gray_normalized = self.ndviImg.astype(float) / 255.0

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
        gray = opencv.cvtColor(self.img, opencv.COLOR_BGR2GRAY)
        _, thresh = opencv.threshold(gray, 0, 255, opencv.THRESH_BINARY_INV + opencv.THRESH_OTSU)

        # noise removal
        kernel = np.ones((10, 10), np.uint8)
        opening = opencv.morphologyEx(thresh, opencv.MORPH_OPEN, kernel, iterations = 2)
        
        # sure background area
        sure_bg = opencv.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = opencv.distanceTransform(opening, opencv.DIST_L2,5)
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

        markers = opencv.watershed(self.img,markers)
        self.img[markers == -1] = [255, 0, 0]

        mask = np.zeros_like(gray)
        mask[markers > 1] = 255
        
        return mask

    def classify_image(self):
        if self.img is None:
            print("Error: Image not loaded properly.")
            return 'unknown'
    
        # Convert image to HSV color space for better color recognition
        hsv_image = opencv.cvtColor(image, opencv.COLOR_BGR2HSV)
        
        # Adjusted HSV color ranges for different environments
        green_range = ((35, 20, 20), (85, 255, 255))  # Rainforest (green, with lower brightness threshold)
        yellow_range = ((15, 40, 40), (35, 255, 255))  # Desert (yellow-tan, more sensitive)
        blue_range = ((85, 30, 30), (135, 255, 255))  # Ocean (expanded blue range)
    
        # Function to calculate percentage of pixels in a given color range
        def calculate_color_percentage(hsv_img, lower_bound, upper_bound):
            mask = opencv.inRange(hsv_img, np.array(lower_bound), np.array(upper_bound))
            color_pixels = opencv.countNonZero(mask)
            total_pixels = hsv_img.shape[0] * hsv_img.shape[1]
            return (color_pixels / total_pixels) * 100
    
        # Calculate the percentage of each color
        green_percentage = calculate_color_percentage(hsv_image, *green_range)
        yellow_percentage = calculate_color_percentage(hsv_image, *yellow_range)
        blue_percentage = calculate_color_percentage(hsv_image, *blue_range)
    
        # Texture Analysis using Local Binary Pattern (LBP)
        gray_image = opencv.cvtColor(image, opencv.COLOR_BGR2GRAY)
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

def globe_build(data):

    # Create the figure
    fig = go.Figure(go.Scattergeo())

    # Update the globe projection and layout
    fig.update_geos(projection_type="orthographic")
    fig.update_layout(height=750, margin={"r":0,"t":50,"l":3,"b":50})
    fig.update_geos(landcolor="#34A56F")
    fig.update_geos(oceancolor="#005477")
    fig.update_geos(showcountries=True)
    fig.update_geos(bgcolor="#000000")
    fig.update_geos(lakecolor="blue")
    fig.update_geos(rivercolor="blue")
    fig.update_geos(riverwidth=2)
    fig.update_geos(showframe=True)
    fig.update_geos(showlakes=True)
    fig.update_geos(showland=True)
    fig.update_geos(showocean=True)
    fig.update_geos(showrivers=True)
    fig.update_geos(showsubunits=True)
    fig.update_geos(lataxis_showgrid=True)
    fig.update_geos(lonaxis_showgrid=True)

    # Create two sets of color keys (legend items)
    strengths = ['Low', 'Medium', 'High']
    colors_red = ['#FFCCCC', '#FF6666', '#FF0000']  # Light to dark red
    colors_purple = ['#E6CCFF', '#A366FF', '#6F00FF']  # Light to dark purple

    # Add red key (bottom left)
    for strength, color in zip(strengths, colors_red):
        fig.add_trace(go.Scattergeo(
            lon=[-180],  # Positioning off the globe
            lat=[-60],   # Bottom left corner
            mode='markers+text',
            marker=dict(size=10, color=color, symbol='circle'),
            text=f"Tree Coverage - {strength}",
            textposition="top center",
            showlegend=True,
            name=f"Tree Coverage - {strength}"
        ))

    # Add purple key (bottom right)
    for strength, color in zip(strengths, colors_purple):
        fig.add_trace(go.Scattergeo(
            lon=[-180],  # Positioning off the globe
            lat=[-60],   # Bottom right corner
            mode='markers+text',
            marker=dict(size=10, color=color, symbol='circle'),
            text=f"NDVI - {strength}",
            textposition="top center",
            showlegend=True,
            name=f"NDVI - {strength}"
        ))

    # Extract latitudes, longitudes, and values
    latitudes = [coord[0][0] for coord in data]
    longitudes = [coord[0][1] for coord in data]
    ndvi_changes = [coord[1] for coord in data]
    tree_coverage_changes = [coord[2] for coord in data]

    # Normalize NDVI changes for color scaling
    ndvi_colorscale = np.array(ndvi_changes)
    min_ndvi = np.min(ndvi_colorscale)
    max_ndvi = np.max(ndvi_colorscale)
    normalized_ndvi = (ndvi_colorscale - min_ndvi) / (max_ndvi - min_ndvi)

    # Normalize Tree Coverage changes for color scaling
    tree_colorscale = np.array(tree_coverage_changes)
    min_tree = np.min(tree_colorscale)
    max_tree = np.max(tree_colorscale)
    normalized_tree = (tree_colorscale - min_tree) / (max_tree - min_tree)

    # Add markers for NDVI (with color bar)
    fig.add_trace(go.Scattergeo(
        lon=longitudes,
        lat=latitudes,
        text=[f"NDVI Change: {ndvi}, Tree Coverage Change: {tree}" for ndvi, tree in zip(ndvi_changes, tree_coverage_changes)],
        mode='markers',
        marker=dict(
            size=[max(5, abs(tree) * 2) for tree in tree_coverage_changes],  # Scale marker size by tree coverage change
            color=normalized_ndvi,  # Use normalized NDVI for color
            colorscale='YlGnBu',  # Colorscale for NDVI
            cmin=0, cmax=1,
            showscale=True,  # Show the NDVI color scale bar
            colorbar=dict(
                title="NDVI Change",
                titleside="right",  # Titleside set to 'right'
                x=0.95,  # Position the color bar on the right
                len=0.6,  # Adjust the length of the color bar
                y=0.5  # Center it vertically
            )
        )
    ))

    # Add markers for Tree Coverage (with separate color bar)
    fig.add_trace(go.Scattergeo(
        lon=longitudes,
        lat=latitudes,
        mode='markers',
        marker=dict(
            size=[max(5, abs(tree) * 2) for tree in tree_coverage_changes],  # Scale marker size by tree coverage change
            color=normalized_tree,  # Use normalized Tree Coverage for color
            colorscale='Reds',  # Colorscale for Tree Coverage
            cmin=0, cmax=1,
            showscale=True,  # Show the Tree Coverage color scale bar
            colorbar=dict(
                title="Tree Coverage Change",
                titleside="right",  # Titleside set to 'right'
                x=-0.05,  # Position the color bar on the left
                len=0.6,  # Adjust the length of the color bar
                y=0.5  # Center it vertically
            )
        ),
        hoverinfo="skip",  # Avoid duplicate hover information
    ))

    # Update layout for the title and legend
    fig.update_layout(
        title={
            'text': "Change in Deforestation (2016-2024)",
            'font': {'size': 24},
            'x': 0.5,  # Center the title
            'y': 0.99,  # Position from the top
            'xanchor': 'center',
            'yanchor': 'top',
        },
        legend=dict(
            title="Strength Keys",
            orientation="h",
            yanchor="top",
            y=-0.1,  # Move legend down to avoid overlapping
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.7)",  # Slightly transparent background
            bordercolor="Black",
            borderwidth=1,
            font=dict(size=12)
        )
    )


    fig.show()



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
    return (SimpleAnalysisTools.getWhitePercentage(imgBundle2.greyScaleSegment()) 
            - SimpleAnalysisTools.getWhitePercentage(imgBundle1.greyScaleSegment()), 
      SimpleAnalysisTools.getMeanNDVIDiff(imgBundle1.returnNdvi(), imgBundle2.returnNdvi()))

folder_path = "images/"
batch_generator = batch_image_loader(folder_path)

output = []
for batch in batch_generator:
    images = load_images(batch)
    final_values = [compare(imgbundle1, imgbundle2) for (imgbundle1, imgbundle2) in images]
    tree_values = [tup[0] for tup in final_values]
    ndvi_values = [tup[1] for tup in final_values]
    coordinates = [((imgbundle1.coords[2] + imgbundle1.coords[0])/2, (imgbundle1.coords[3] + imgbundle1.coords[1])/2) for (imgbundle1, imgbundle2) in images]
    for result in zip(coordinates, tree_values, ndvi_values):
        output.append(result)

globe_build(output)

