# 🚀 Deep Learning Projects 
## Project: Emotion detection (FER-2013 dataset)

The dataset consists of 35,685 examples of gray-scale images of faces with dimensions of 48x48 pixels. These images are divided into a training dataset and a test dataset. The images are categorized based on the emotions depicted in the facial expressions. The emotion categories include:

Targets/labels: The target classes (the emotion categories)
Each class corresponds to a specific emotion category. The emotion categories given above.
- 🤗 Happiness
- 🧑 Neutral
- 😟 Sadness
- 😠 Anger
- 😯 Surprise


- Features: The images
The images serve as the features or inputs to the model. These images contain visual information that the model will use to make predictions about the corresponding emotions.
the gray-scale images of faces with dimensions of 48x48 pixels. Each pixel in the image represents a feature value. The dataset consists of 35,685 examples of these images.

 Performance and accuracies:
 I explored various approaches to observe the outcomes and enhance the model's performance.
- Highest training accuracy : 78%
- Highest validation accuracy : 77%

Visualization:
<p float="left">
<img src="https://github.com/Abdullah-TU/Deep-Learning-Projects/blob/main/emotion_viz.png" width="1000" height="500">
</p>

# Project: Text-to-text translation
## Target of the project:
Our goal is to develop a deep learning model for translating different languages. However, our current focus is on English to Finnish translation.

## About Dataset
Dataset collected from Tatoeba repository https://tatoeba.org/en/downloads . it consists of two files :
- eng.txt: Contains Enlish sentences in each lines. Total: 100248 English sentenes
- fin.txt: Contains FInnish sentences in each lines. Total: 100248 Finnish sentenes

### Identify the features and the targets:
- 🍃 Features: Source Language Text (English sentences)
- 🎯 Targets/labels: Target Language Text (Finnish sentences)

### Translation using the model:
- **Original (data)   : you have to speak french here**
- Target (data)     : sinun täytyy puhua täällä ranskaa
- Translated (model): teidän täytyy puhua ranskaa täällä
&nbsp;
- **Original (data)   : its against my principles**
- Target (data)     : se on vastoin minun periaatteitani
- Translated (model): se on minun periaatteitani vastaan
&nbsp;
- **Original (data)   : she expressed her thanks for the present**
- Target (data)     : hän ilmaisi kiitollisuutensa lahjasta
- Translated (model): hän pohti kysymystä hetkisen
&nbsp;
- **Original (data)   : i used to have a minidisc player**
- Target (data)     : minulla oli aikaisemmin minidiscsoitin
- Translated (model): minulla oli aikaisemmin minidiscsoitin
&nbsp;
- **Original (data)   : i m not dating anyone**
- Target (data)     : en seurustele kenenkään kanssa tällä hetkellä
- Translated (model): en ole varma
  &nbsp;

# Project: Face Spoof Detection 👤🕵️‍♂️ 
The dataset encompasses 42,000 examples of gray-scale images of faces, each with dimensions of 64x64 pixels. These images have been segregated into training, validation, and test sets for model development and evaluation.
### Dataset Characteristics:
- Image Specifications: Gray-scale images measuring 64x64 pixels.
- Categories: Each image is categorized as either a genuine or spoofed facial image.
  
### Targets/Labels:
The target labels indicate whether an image belongs to the genuine or spoofed category, essential for classification purposes. The dataset distinguishes between these two classes to enable accurate model predictions.
#### Performance and Evaluation:
Throughout the project, several methodologies were employed to refine and enhance the model's performance.
- Training Accuracy: The model achieved its highest training accuracy at 78%.
- Validation Accuracy: Demonstrating robustness, the highest validation accuracy reached 77%.
- Test Accuracy: 67.3%
- Test Precision: 67.3%
- Test Recall: 95.3%
- Test F1-score: 78.9%
  
<p float="left">
<img src="https://github.com/Abdullah-TU/Deep-Learning-Projects/blob/main/confusion_spoof.png" width="400" height="300">
<img src="https://github.com/Abdullah-TU/Deep-Learning-Projects/blob/main/metric_spoof.png" width="400" height="300">
</p>

Visualization:
<p float="left">
<img src="https://github.com/Abdullah-TU/Deep-Learning-Projects/blob/main/face_result_spoof.png" width="1000" height="320">
</p>

# Project: Semantic Segmentation for Self-Driving Cars

**Objective:** Develop a semantic segmentation model using deep learning techniques to accurately label each pixel in images captured from the CARLA self-driving car simulator. The model aims to categorize pixels into classes like cars, roads, and other objects, contributing to better environmental understanding for self-driving car systems.

**Dataset:**
- The dataset contains RGB images and corresponding labeled semantic segmentations.
- Captured using the CARLA self-driving car simulator.
- Labels include objects like cars, roads, and other elements in the scene.

**Approach:**
1. **Dataset Preparation:**
   - Data loading and preprocessing, including image resizing and mask conversion.


2. **Model Architecture:**
   - Utilize a U-Net architecture, which includes both downsampling and upsampling pathways.
   - Convolutional downsampling blocks capture features.
   - Convolutional upsampling blocks recover spatial information.

3. **Model Compilation and Training:**
   - Compile the model with 'adam' optimizer and sparse categorical cross-entropy loss.
   - Train the model on the prepared dataset for a specified number of epochs.
   - Monitor the accuracy and loss metrics during training.

4. **Model Evaluation and Visualization:**
   - Display sample predictions on the test dataset.
   - Utilize the trained model to predict segmentations on new images.
   - Compare the input image, true mask, and predicted mask for visual assessment.

**Libraries and Tools:**
- TensorFlow and Keras for model building, training, and evaluation.
- TensorFlow Datasets for creating input pipelines.
- Data preprocessing using various transformations.
- U-Net architecture for semantic segmentation.
- Metrics such as accuracy and loss for model evaluation.
- Visualization using matplotlib and imageio.

**Results and Future Work:**
- The model demonstrates improvement in accurately segmenting objects in the scene.
- Further fine-tuning and experimentation with hyperparameters could enhance performance.
- The trained model can be integrated into self-driving car systems to aid in environmental perception.

<p float="left">
<img src="https://github.com/Abdullah-TU/Images-for-Other-Files/blob/ab483a918cf96ca8fff0c2a6433f83665ea3d7ea/download.png" width="1000" height="300">
<img src="https://github.com/Abdullah-TU/Images-for-Other-Files/blob/ab483a918cf96ca8fff0c2a6433f83665ea3d7ea/download%20(2).png" width="1000" height="300">
<img src="https://github.com/Abdullah-TU/Images-for-Other-Files/blob/ab483a918cf96ca8fff0c2a6433f83665ea3d7ea/download%20(3).png" width="1000" height="300">
<img src="https://github.com/Abdullah-TU/Images-for-Other-Files/blob/ab483a918cf96ca8fff0c2a6433f83665ea3d7ea/download%20(4).png" width="1000" height="300">
</p>

# Project:Neural networks(CNN), CIFAR -10 dataset ( Exercise 4)

Project is about image classification using a Convolutional Neural Network (CNN) on the CIFAR-10 dataset.

Here's a summary of Project:

1. **Data Loading and Preprocessing**:
   - The code starts with loading the CIFAR-10 dataset using Keras' `cifar10.load_data()` function.
   - The pixel values of the images are normalized to the range [0, 1].
   - The class labels are one-hot encoded.

2. **CNN Model Construction**:
   - A Sequential model is created using Keras.
   - The model consists of a series of Convolutional, Batch Normalization, MaxPooling, Dropout, and Fully Connected layers.
   - Three sets of Convolutional layers are used, each followed by Batch Normalization and MaxPooling, with increasing filter counts.
   - The Fully Connected layers consist of two Dense layers with ReLU activation, Batch Normalization, and Dropout.
   - The output layer has 10 neurons with softmax activation for multi-class classification.

3. **Model Compilation**:
   - The model is compiled using the Adam optimizer and categorical cross-entropy loss function.
   - The accuracy metric is specified for evaluation during training.

4. **Data Augmentation and Training**:
   - Image data augmentation is performed using Keras' `ImageDataGenerator`, which randomly applies zoom and horizontal flip to augment the training data.
   - The model is trained using the augmented data with a specified number of epochs and batch size.

5. **Training Progress Visualization**:
   - Training and validation accuracy/loss values are plotted using Matplotlib to visualize the model's learning progress over epochs.

6. **Prediction and Accuracy Calculation**:
   - The trained model is used to predict class labels for the test dataset.
   - The predicted labels are converted from one-hot encoded form to integer labels.
   - The accuracy of the model is calculated by comparing the predicted labels with the true labels.

7. **Performance of the model**:
   - The calculated accuracy is printed out, indicating the performance of the CNN on the test dataset.
   - The CNN achieved approximately 85% accuracy, surpassing the accuracy of 1-NN and Bayes classifiers.
