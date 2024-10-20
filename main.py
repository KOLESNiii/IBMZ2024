from lawrence import batchImages
import imageBundle
import segmentsplit
import classification
import timk
import analysis

if __name__ == "__main__":
    for (img16, img24) in zip(batchImages('images2016'), batchImages('images2024')):
        biome16 = classification.classify_image(img16)
        biome24 = classification.classify_image(img24)
        
        analysis.compare(img16, img24)