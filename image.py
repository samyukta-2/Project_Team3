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

















































































class ImageForgeryDetector:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    def prepare_dataset(self, folder):
        images, labels = load_images_from_folder(folder)
        features = []
        extended_labels = []


        for img, label in zip(images, labels):
            combined_feat = np.hstack([
                extract_lbp_features(img),
                extract_color_histogram(img),
                extract_edge_histogram(img)
 ])
            features.append(combined_feat)
                        extended_labels.append(label)
            for aug_img in augment_image(img):
                aug_feat = np.hstack([
                    extract_lbp_features(aug_img),
extract_color_histogram(aug_img),
                    extract_edge_histogram(aug_img)
                ])
                features.append(aug_feat)
                extended_labels.append(label)
        return np.array(features), np.array(extended_labels)



    def train(self, folder):
   X, y = self.prepare_dataset(folder)
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)
        self.classifier.fit(X_train, y_train)















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
        elif args.test:
        detector.load_model(args.model)
        result = detector.predict(args.test)
        print(f"Prediction for {args.test}: {'Forged' if result == 'forged' else 'Authentic'}")
    elif args.cross_validate:
        detector.cross_validate(args.train, cv=args.cross_validate)

if __name__ == '__main__':
    main()

def extract_lbp_features(image, radius=3, n_points=24):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-7)
        return hist


def augment_image(image):
augmented_images = [
        cv2.flip(image, 1),
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(image, cv2.ROTATE_180),
        cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]
    return augmented_images


         y_pred = self.classifier.predict(X_test)
         print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
        print(classification_report(y_test, y_pred))
        self.plot_confusion_matrix(y_test, y_pr
