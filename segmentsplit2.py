import math

# Constants
earth_radius = 6371.0  # Earth radius in km
km_per_deg_lat = 111  # Kilometers per degree of latitude

# Function to calculate the longitude step at a given latitude
def calculate_longitude_step(lat, side_km):
    km_per_deg_lon = 111 * math.cos(math.radians(lat))  # Kilometers per degree of longitude at latitude
    return side_km / km_per_deg_lon


# Generate grid from bottom-left (-90, -180) to top-right (90, 180)
def get_all_grids(side_length_km = 85):
    # Latitude step (constant since km per degree latitude is constant)
    lat_step = side_length_km / km_per_deg_lat
    lat = -90
    count = 0
    while lat + lat_step <= 90:
        lon = -180
        while lon + calculate_longitude_step(lat, side_length_km) <= 180:
            lon_step = calculate_longitude_step(lat, side_length_km)
            
            # Bottom-left and top-right corners of the box
            bottom_left = [lat, lon]
            top_right = [lat + lat_step, lon + lon_step]
            
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
    lat = -90
    count = 0
    while lat + lat_step <= 90:
        lon = -180
        while lon + calculate_longitude_step(lat, side_length_km) <= 180:
            lon_step = calculate_longitude_step(lat, side_length_km)
            if count == n:
                return [lat, lon, lat + lat_step, lon + lon_step]

            lon += lon_step
            count += 1
        # Move to the next box in latitude
        lat += lat_step