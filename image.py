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

 rows, cols = image.shape[:2]
    pts1 = np.float32([[50,50], [200,50], [50,200]])
    pts2 = np.float32([[10,100], [200,50], [100,250]])
    M = cv2.getAffineTransform(pts1, pts2)
    augmented.append(cv2.warpAffine(image, M, (cols, rows)))

    return augmented


def plot_feature_importance(self, feature_names=None):
    
    importances = self.classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(20), importances[indices[:20]], align='center')
    plt.xticks(range(20), indices[:20])
    plt.xlabel('Feature Index')
    plt.ylabel('Relative Importance')
    plt.show()

ImageForgeryDetector.plot_feature_importance = plot_feature_importance

def tune_hyperparameters(self, folder, param_grid=None):
    
    from sklearn.model_selection import GridSearchCV

    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

    X, y = self.prepare_dataset(folder)
    grid_search = GridSearchCV(self.classifier, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X, y)

    self.classifier = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.2f}")

ImageForgeryDetector.tune_hyperparameters = tune_hyperparameters

def plot_roc_curve(self, folder):
    
    from sklearn.metrics import RocCurveDisplay

    X, y = self.prepare_dataset(folder)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    self.classifier.fit(X_train, y_train)

    RocCurveDisplay.from_estimator(self.classifier, X_test, y_test)
    plt.title('ROC Curve')
    plt.show()

ImageForgeryDetector.plot_roc_curve = plot_roc_curve


def main():
    parser = argparse.ArgumentParser(description='Image Forgery Detector')
    parser.add_argument('--train', metavar='DATASET_DIR', help='Train using dataset directory')
    parser.add_argument('--test', metavar='IMAGE_PATH', help='Test single image')
    parser.add_argument('--model', default='model.pkl', help='Model file path')
    parser.add_argument('--cross-validate', type=int, help='Run cross-validation with K folds')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--features', action='store_true', help='Show feature importance plot')
    parser.add_argument('--roc', action='store_true', help='Generate ROC curve')
    args = parser.parse_args()

    detector = ImageForgeryDetector()

    if args.train:
        if args.tune:
            detector.tune_hyperparameters(args.train)
        else:
            detector.train(args.train)
        detector.save_model(args.model)

        if args.features:
            detector.plot_feature_importance()
        if args.roc:
            detector.plot_roc_curve(args.train)

def predict(self, image_path):
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Invalid image format")

        if img.shape[2] != 3:
            raise ValueError("Image must have 3 color channels")

        features = np.hstack([
            extract_lbp_features(img),
            extract_color_histogram(img),
            extract_edge_histogram(img),
            extract_ela_features(img)
        ])
        return self.classifier.predict([features])[0]

    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

ImageForgeryDetector.predict = predict




