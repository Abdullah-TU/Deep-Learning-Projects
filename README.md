# 🚀 Deep Learning Projects 
## Project: Emotion detection (FER-2013 dataset)

The dataset consists of 35,685 examples of gray-scale images of faces with dimensions of 48x48 pixels. These images are divided into a training dataset and a test dataset. The images are categorized based on the emotions depicted in the facial expressions. The emotion categories include:
- 🤗 Happiness
- 🧑 Neutral
- 😟 Sadness
- 😠 Anger
- 😯 Surprise
- 😑 Disgust
- 😨 Fear

Targets/labels: The target classes (the emotion categories)
Each class corresponds to a specific emotion category. The emotion categories include:
- 🤗 Happiness
- 🧑 Neutral
- 😟 Sadness
- 😠 Anger
- 😯 Surprise
- 😑 Disgusted
- 😨 Fear
  
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
- Original (data)   : you have to speak french here
- Target (data)     : sinun täytyy puhua täällä ranskaa
- Translated (model): teidän täytyy puhua ranskaa täällä
  
&nbsp;
- Original (data)   : its against my principles
- Target (data)     : se on vastoin minun periaatteitani
- Translated (model): se on minun periaatteitani vastaan
  
&nbsp;
- Original (data)   : she expressed her thanks for the present
- Target (data)     : hän ilmaisi kiitollisuutensa lahjasta
- Translated (model): hän pohti kysymystä hetkisen
  
&nbsp;
- Original (data)   : i used to have a minidisc player
- Target (data)     : minulla oli aikaisemmin minidiscsoitin
- Translated (model): minulla oli aikaisemmin minidiscsoitin
  
  &nbsp;

- Original (data)   : i m not dating anyone
- Target (data)     : en seurustele kenenkään kanssa tällä hetkellä
- Translated (model): en ole varma
  &nbsp;

# Project Description: Semantic Segmentation for Self-Driving Cars

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
