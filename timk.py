import requests
from math import radians, cos, sin, sqrt, atan2
from concurrent.futures import ThreadPoolExecutor, as_completed

import math
#Secrets
client_id = '9b40331a-7b0e-49cd-ab64-9c908c74e538'
client_secret = 'lu7DmSlKKu8FvVPEry7tPK4DTIJt2dlQ'

RESOLUTION = 25

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
    lat1 = coords[0]
    lon1 = coords[1]
    lat2 = coords[2]
    lon2 = coords[3]
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
    if photo:
        with open(filename, "wb") as f:
            f.write(photo)
        print("Photo saved")
    else:
        print("No photo to save")

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
                "width": RESOLUTION * bbox_width,
                "height": RESOLUTION * bbox_height,
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
            if response.status_code != 200:
                print(f"Error getting image {imageNum} due to status code {response.status_code}")
                return headers, url, imageNum, bbox, year
            else:
                if evalscript == colour_evalscript:
                    save_photo(response.content, f"{year}img{imageNum}.jpeg")
                else:
                    save_photo(response.content, f"NDVI{year}img{imageNum}.jpeg")
    except Exception as e:
        print(f"Error getting image {imageNum}")
        #print(e)
        return headers, url, imageNum, bbox, year
    return True

def download_images(year):
    token = getNewToken()
    if not token:
        return
    url = "https://services.sentinel-hub.com/api/v1/process"
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}"
    }

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(post_request, headers, url, n, bbox, year): (n,bbox) for n, bbox in get_all_grids()}
        bad_requests = []
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
                if data != True:
                    bad_requests.append(data)
            except Exception as exc:
                print(f"Error getting image {url}: {exc}")
            else:
                print(f"Image {url} downloaded")

    
download_images(2024)
token = getNewToken()