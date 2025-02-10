# 🚀 Deep Learning Projects 
## Project: Emotion detection (FER-2013 dataset)
The dataset consists of 35,685 examples of gray-scale images of faces with dimensions of 48x48 pixels. These images are divided into a 
training dataset and a test dataset. 
The images are categorized based on the emotions depicted in 
the facial expressions. The emotion categories include:
- Sad 😢
- Neutral 😐
- Disgust 🤢


Targets/labels: The target classes (the emotion categories)
Each class corresponds to a specific emotion category. The emotion categories given above.
- Test Accuracy: 67.3%
- Test Precision: 67.3%
- Test Recall: 95.3%
- Test F1-score: 78.9%

- Features: The images
The images serve as the features or inputs to the model. These images contain visual information that the model will use to make predictions about the 
corresponding emotions.
the gray-scale images of faces with dimensions of 48x48 pixels. Each pixel in the image represents a feature value. The dataset consists of 35,685 examples 
of these images.
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

## 7. Performance of the Model

- The calculated accuracy is printed out, indicating the performance of the CNN on the test dataset.

### Model Accuracy: The CNN achieved an accuracy of approximately **85%**

![Model Accuracy Screenshot](https://github.com/user-attachments/assets/a0e72a04-8bd4-4a69-866e-d3da261dade7)

## Visualization

Demonstrating the results of the CNN model:

<img width="266" alt="Visualization 1" src="https://github.com/user-attachments/assets/4ee3df15-e427-4154-b9db-fcb0fd67cb0a">
<img width="260" alt="Visualization 2" src="https://github.com/user-attachments/assets/d455a037-9427-4bc1-86ba-d46314c773b3">
<img width="266" alt="Visualization 3" src="https://github.com/user-attachments/assets/2d186aee-17cf-4b8a-ae92-c8be83a8c605">
<img width="266" alt="Visualization 4" src="https://github.com/user-attachments/assets/63f7a26e-17d3-4248-bd5a-4ea6c5ccf025">
<img width="263" alt="Visualization 5" src="https://github.com/user-attachments/assets/36a27728-1e5f-4b9f-a560-c83ae091bc47">
<img width="272" alt="Visualization 6" src="https://github.com/user-attachments/assets/995ece26-40dd-476b-a999-9c1c595bf500">


# 🧠 Project: Recurrent Neural Network (RNN) for Text Classification

## 🎯 Project Goal
The objective of this project is to develop a Recurrent Neural Network (RNN) model to classify text data. We used the popular IMDB movie reviews dataset to build a model that can predict the sentiment (positive or negative) of a given review.

## 📊 About the Dataset
The dataset used in this project is the **IMDB Reviews Dataset** provided by TensorFlow. It contains **50,000 movie reviews** split evenly between training and testing datasets.

### Data Details:
- **Training data**: 25,000 labeled movie reviews (positive/negative).
- **Test data**: 25,000 labeled movie reviews (positive/negative).
- **Feature**: Sequence of words (movie review text).
- **Label**: Sentiment (1 for positive, 0 for negative).

## 🛠️ Implementation Steps
This project was implemented using TensorFlow and Keras. Below is a summary of the workflow:

### Step 1: Data Loading and Preprocessing
- Loaded the **IMDB dataset** using TensorFlow's `imdb.load_data()` function.
- Tokenized and converted the reviews into sequences of integers.
- Applied **padding** to ensure uniform sequence lengths.

### Step 2: Building the RNN Model
- Created a sequential model with the following layers:
  - **Embedding Layer**: Converts integer sequences into dense vector embeddings.
  - **Simple RNN Layer**: Captures sequential dependencies in the text data.
  - **Dense Layer**: Used for output classification (binary sentiment analysis).

### Step 3: Compiling and Training the Model
- Used the **binary cross-entropy** loss function for binary classification.
- Optimized using the **Adam optimizer**.
- Trained the model with **10 epochs** and a batch size of **32**.

### Step 4: Model Evaluation
- Evaluated the model performance on the test dataset.
- Achieved an accuracy of around **85-87%** on the test set.

## 📈 Results and Analysis
Here are the key outcomes:

- **Training Accuracy**: High accuracy on the training data, demonstrating that the model effectively learned patterns in the dataset.
- **Validation Accuracy**: Consistent performance on validation data, indicating good generalization.
- **Test Accuracy**: Achieved an accuracy of approximately **83.76%** on unseen test data, suggesting the model performs well in predicting sentiment.

### Predicted vs. Actual Sentiments for 10 Test Samples:

Review 1:  
**Predicted Sentiment**: Positive  
**Actual Sentiment**: Negative  
--------------------------------------------------  
Original review:  
? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? please give this one a miss br br ? ? and the rest of the cast rendered terrible performances the show is flat flat flat br br i don't know how michael madison could have allowed this one on his plate he almost seemed to know this wasn't going to work out and his performance was quite ? so all you madison fans give this a miss.

Review 2:  
**Predicted Sentiment**: Positive  
**Actual Sentiment**: Positive  
--------------------------------------------------  
Original review:  
psychological ? it's very interesting that robert altman directed this considering the style and structure of his other films still the trademark altman audio style is evident here and there i think what really makes this film work is the brilliant performance by sandy dennis it's definitely one of her darker characters but she plays it so perfectly and convincingly that it's scary michael burns does a good job as the mute young man regular altman player michael murphy has a small part the ? moody set fits the content of the story very well in short this movie is a powerful study of loneliness sexual ? and desperation be patient ? up the atmosphere and pay attention to the wonderfully written script br br i praise robert altman this is one of his many films that deals with unconventional fascinating subject matter this film is disturbing but it's sincere and it's sure to ? a strong emotional response from the viewer if you want to see an unusual film some might even say bizarre this is worth the time br br unfortunately it's very difficult to find in video stores you may have to buy it off the internet

Review 3:  
**Predicted Sentiment**: Positive  
**Actual Sentiment**: Positive  
--------------------------------------------------  
Original review:  
everyone's horror the ? promptly eats the mayor and then goes on a merry rampage ? citizens at random a title card ? reads news of the king's ? throughout the kingdom when the now terrified ? once more ? ? for help he loses his temper and ? their community with lightning ? the moral of our story delivered by a hapless frog just before he is eaten is let well enough alone br br considering the time period when this startling little film was made and considering the fact that it was made by a russian ? at the height of that ? country's civil war it would be easy to see this as a ? about those events ? may or may not have had ? turmoil in mind when he made ? but whatever ? his choice of material the film stands as a ? tale of universal ? ? could be the soviet union italy germany or japan in the 1930s or any country of any era that lets its guard down and is overwhelmed by ? it's a fascinating film even a charming one in its macabre way but its message is no joke

Review 4:  
**Predicted Sentiment**: Positive  
**Actual Sentiment**: Negative  
--------------------------------------------------  
Original review:  
? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? i generally love this type of movie however this time i found myself wanting to kick the screen since i can't do that i will just complain about it this was absolutely idiotic the things that happen with the dead kids are very cool but the alive people are absolute idiots i am a grown man pretty big and i can defend myself well however i would not do half the stuff the little girl does in this movie also the mother in this movie is reckless with her children to the point of neglect i wish i wasn't so angry about her and her actions because i would have otherwise enjoyed the flick what a number she was take my advise and fast forward through everything you see her do until the end also is anyone else getting sick of watching movies that are filmed so dark anymore one can hardly see what is being filmed as an audience we are ? involved with the actions on the screen so then why the hell can't we have night vision

Review 5:  
**Predicted Sentiment**: Positive  
**Actual Sentiment**: Positive  
--------------------------------------------------  
Original review:  
? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? like some other people wrote i'm a die hard mario fan and i loved this game br br this game starts slightly boring but trust me it's worth it as soon as you start your hooked the levels are fun and ? they will hook you ? your mind turns to ? i'm not kidding this game is also ? and is beautifully done br br to keep this spoiler free i have to keep my mouth shut about details but please try this game it'll be worth it br br story 9 9 action 10 1 it's that good ? 10 attention ? 10 average 10 

## 🔗 Resources
- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
