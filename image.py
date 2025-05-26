import cv2
import numpy as np
import os
import glob
import joblib
import argparse
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
         if os.path.isdir(label_folder):
             for filename in glob.glob(os.path.join(label_folder, '*.jpg')):
                 img = cv2.imread(filename, cv2.IMREAD_COLOR)
                 if img is not None:
                     images.append(img)
                     labels.append(label)
return images, labels








































































































































































def calculate_ela(image, quality=95):
    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    temp_image = cv2.imread(temp_path)
    os.remove(temp_path)
    ela = cv2.absdiff(image, temp_image)
    return ela.flatten()

def extract_ela_features(image):
    ela_image = calculate_ela(image)
    hist, _ = np.histogram(ela_image, bins=256, range=(0, 256))
    hist = hist.astype('float') / (hist.sum() + 1e-7)
    return hist

def augment_image(image):
    
    augmented = [image]

    
    augmented.append(cv2.flip(image, 1))
    augmented.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = hsv[..., 2] * 0.8  
    augmented.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

    gauss = np.random.normal(0, 0.1*255, image.shape).astype('uint8')
    augmented.append(cv2.add(image, gauss))

