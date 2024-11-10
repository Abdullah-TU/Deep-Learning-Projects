# 🚀 Deep Learning Projects 
## Project: Emotion detection (FER-2013 dataset)
The dataset consists of 35,685 examples of gray-scale images of faces with dimensions of 48x48 pixels. These images are divided into a training dataset and a test dataset. 
The images are categorized based on the emotions depicted in 
the facial expressions. The emotion categories include:
- Happy 😊
- Sad 😢
- Neutral 😐
- Disgust 🤢
- Angry 😠
- Surprise 😮
  
Targets/labels: The target classes (the emotion categories)
Each class corresponds to a specific emotion category. The emotion categories given above.


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

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

### Step 4: Model Evaluation
- Evaluated the model performance on the test dataset.
- Achieved an accuracy of around **85-87%** on the test set.

---

## 📈 Results and Analysis
Here are the key outcomes:

- **Training Accuracy**: High accuracy on the training data, demonstrating that the model effectively learned patterns in the dataset.
- **Validation Accuracy**: Consistent performance on validation data, indicating good generalization.
- **Test Accuracy**: Achieved an accuracy of approximately **87%** on unseen test data, suggesting the model performs well in predicting sentiment.

### Loss and Accuracy Plots
You can refer to the notebook for detailed plots showing the training and validation loss and accuracy over the epochs.

---

## 📝 Usage Instructions
To replicate the results or experiment with the code, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rnn-text-classification.git
   cd rnn-text-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook RNN_Basics.ipynb
   ```

---

## 📂 Explore the Notebook
The full implementation, including preprocessing, model building, training, and evaluation, can be found in the [notebook file](./RNN_Basics.ipynb).

---

## 🔗 Resources
- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/guide)
- [Understanding RNNs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

Feel free to adjust any sections to better fit your project specifics. Let me know if you need further customization or additional details!  
  
