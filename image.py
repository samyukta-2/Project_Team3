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












































































































    def predict(self, image_path):
img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
            features = np.hstack([
            extract_lbp_features(img),
            extract_color_histogram(img),
            extract_edge_histogram(img)
        ])
        return self.classifier.predict([features])[0]


def plot_confusion_matrix(self, y_true, y_pred):
cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                  xticklabels=['Forged', 'Real'],
                  yticklabels=['Forged', 'Real'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        def cross_validate(self, folder, cv=5):
             X, y = self.prepare_dataset(folder)
        scores = cross_val_score(self.classifier, X, y, cv=cv)
        print(f'Cross-validation scores: {scores}')
        print(f'Mean accuracy: {scores.mean():.2f}')

def save_model(self, path='forgery_detector.pkl'):
            joblib.dump(self.classifier, path)
        print(f'Model saved to {path}')

def load_model(self, path='forgery_detector.pkl'):
     self.classifier = joblib.load(path)
        print(f'Model loaded from {path}')

        def main():
            parser = argparse.ArgumentParser(description='Image Forgery Detector')
    parser.add_argument('--train', metavar='DATASET_DIR', help='Train using dataset directory')
    parser.add_argument('--test', metavar='IMAGE_PATH', help='Test single image')
     parser.add_argument('--model', default='model.pkl', help='Model file path')
    parser.add_argument('--cross-validate', type=int, help='Run cross-validation with K folds')
    args = parser.parse_args()

    detector = ImageForgeryDetector()

    if args.train:
        detector.train(args.train)
        detector.save_model(args.model)
