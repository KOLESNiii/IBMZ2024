import requests
from math import radians, cos, sin, sqrt, atan2
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from time import sleep
import logging
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import datetime

from segmentsplit import get_grids

#Secrets
client_id = '30fa2c14-8e3b-42bf-a6e3-85fc5dadef2c'
client_secret = 'swd5WcVv4bPCJfF2OvXLoT5b5gNd6JuT'

RESOLUTION = 100
sizeInKM = ((2500 * RESOLUTION) // 10000) * 9
actualResolution = sizeInKM * 1000 / 2500

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def getNewToken():
    token_url = 'https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token'
    token_header = {'content-type': 'application/x-www-form-urlencoded'}
    token_data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }
    response = requests.post(url=token_url, headers=token_header, data=token_data)
    if response.status_code != 200:
        print('Error getting token')
        return None
    return response.json()['access_token']

def getDistanceBetweenCoords(coords):
    lon1 = coords[0]
    lat1 = coords[1]
    lon2 = coords[2]
    lat2 = coords[3]
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance

def save_photo(photo, filename):
    logger.info("Saving photo")
    if photo:
        with open(filename, "wb") as f:
            f.write(photo)
        logger.info("Photo saved")
    else:
        logger.warning("No photo to save")

def post_request(headers, url, imageNum :int, bbox : list[int], year : int) -> None:
    bbox_width = getDistanceBetweenCoords([bbox[0], bbox[1], bbox[0], bbox[3]])
    bbox_height = getDistanceBetweenCoords([bbox[0], bbox[1], bbox[2], bbox[1]])
    colour_evalscript = "//VERSION=3\n\nlet minVal = 0.0;\nlet maxVal = 0.4;\n\nlet viz = new HighlightCompressVisualizer(minVal, maxVal);\n\nfunction evaluatePixel(samples) {\n    let val = [samples.B04, samples.B03, samples.B02];\n    val = viz.processList(val);\n    val.push(samples.dataMask);\n    return val;\n}\n\nfunction setup() {\n  return {\n    input: [{\n      bands: [\n        \"B02\",\n        \"B03\",\n        \"B04\",\n        \"dataMask\"\n      ]\n    }],\n    output: {\n      bands: 4\n    }\n  }\n}\n\n"
    ndvi_evalscript = "//VERSION=3\n\nlet viz = new HighlightCompressVisualizerSingle();\n\nfunction evaluatePixel(samples) {\n    let val = index(samples.B08, samples.B04);\n    val = viz.process(val);\n    val.push(samples.dataMask);\n    return val;\n}\n\nfunction setup() {\n  return {\n    input: [{\n      bands: [\n        \"B04\",\n        \"B08\",\n        \"dataMask\"\n      ]\n    }],\n    output: {\n      bands: 2\n    }\n  }\n}\n\n"

    try:
        for evalscript in [colour_evalscript, ndvi_evalscript]:
            data = {
            "input": {
                "bounds": {
                "bbox": bbox
                },
                "data": [
                {
                    "dataFilter": {
                    "timeRange": {
                        "from": f"{year}-05-02T00:00:00Z",
                        "to": f"{year}-10-19T23:59:59Z"
                    },
                    "maxCloudCoverage": 3
                    },
                    "type": "sentinel-2-l1c"
                }
                ]
            },
            "output": {
                "width": bbox_width * 1000 / actualResolution,
                "height": bbox_height * 1000 / actualResolution,
                "responses": [
                {
                    "identifier": "default",
                    "format": {
                    "type": "image/jpeg"
                    }
                }
                ]
            },
            "evalscript": evalscript
            }

            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 400:
                logger.error(f"Error 400:  Width: {actualResolution * bbox_width}, Height: {actualResolution * bbox_height}")
                logger.debug(response.content)
                logger.debug(imageNum)
                logger.debug(bbox)
            elif response.status_code == 429:
                logger.info("Rate limit exceeded")
                sleep(1)
                return headers, url, imageNum, bbox, year
            elif response.status_code == 403:
                logger.error("Error 403: Forbidden")
                logger.debug(response.content)
                raise PermissionError(f"Error 403 forbidden on item {imageNum}")
                return headers, url, imageNum, bbox, year
            elif response.status_code != 200:
                logger.error(f"Error getting image {imageNum} due to status code {response.status_code}")
                logger.debug(response.content)
                return headers, url, imageNum, bbox, year
            else:
                if evalscript == colour_evalscript:
                    save_photo(response.content, f"actualImagesLowRes/{year}img{imageNum}.jpeg")
                else:
                    save_photo(response.content, f"actualImagesLowRes/NDVI{year}img{imageNum}.jpeg")
    except Exception as e:
        logger.error(f"Error getting image {imageNum}")
        logger.error(e)
        return headers, url, imageNum, bbox, year
    return True

def download_images(year):
    token = getNewToken()
    if not token:
        logger.error("Error getting token")
        return
    url = "https://services.sentinel-hub.com/api/v1/process"
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}"
    }
    num = 25010
    count = 0 
    for i in range(13, 20):
        first = i * 500
        last = (i + 1) * 500 - 1
        with ThreadPoolExecutor(max_workers=10) as executor:
            logger.info("Starting threads")
            futures = [executor.submit(post_request, headers, url, n, bbox, year) for n, bbox in get_grids(sizeInKM, first=first, last=last)]
            wait(futures)
            results = [future.result() for future in futures]
            logger.info(results)
            bad_requests = [result for result in results if result != True]
            logger.info("Finished threads")
            logger.warning(f"Bad requests: {len(bad_requests)}")
            logger.warning(f"Bad requests: {[result[2] for result in bad_requests]}")
            with open(f"bad_requests_{year}.txt", "a") as f:
                f.write(", ".join([str(result[2]) for result in bad_requests]) + ",")

download_images(2024)
#logger.info("Running")

