# Image Classification with Random Forest (GridSearchCV)

## Overview
This project uses Random Forest to classify images into 5 categories from `images.zip`. The notebook loads images from class folders, resizes them to 64×64, normalizes pixel values, flattens RGB images into feature vectors (64×64×3 = 12288 features), trains a Random Forest model using GridSearchCV, evaluates it using accuracy/classification report and a confusion matrix, visualizes feature importance, and predicts the class of a new image.

## Dataset
- Input file: `images.zip`
- After extraction, the dataset must contain one folder per class (folder name = class label), e.g.:
  dalmation, dollar_bill, pizza, soccer_ball, sunflower

## How to Run (Google Colab)
1. Upload `images.zip` to Google Colab.
2. Extract the zip file.
3. Run the notebook cells in order.

## Outputs
- Best hyperparameters from GridSearchCV
- Test accuracy
- Classification report (precision, recall, F1-score)
- Confusion matrix plot
- Feature importance plot
- Predicted class for a new image using the same preprocessing steps

## Notes
Prediction on new images must use the exact same preprocessing (resize → normalize → flatten) as training to avoid shape/scale mismatches.
