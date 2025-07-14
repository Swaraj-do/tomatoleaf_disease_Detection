Dataset used:=https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf

1. Problem Statement¶
- Given a dataset of tomato leaf images, the task is to develop a machine learning model that can accurately classify the images into different disease categories. The goal is to help farmers quickly identify and treat diseased tomato plants, thereby improving crop yield and reducing losses.
2. Our Working
We are using a dataset containing images of tomato leaves affected by various diseases. Our approach involves training a deep learning model on this dataset to classify each image into one of the disease categories. By analyzing the patterns and features in the images, the model can learn to differentiate between healthy and diseased leaves, aiding in the early detection and management of plant diseases.

1. Dataset Description:¶
- We are using a dataset containing images of tomato leaves affected by various diseases. Each image is labeled with the corresponding disease category.Training dataset contain 10000 labeled images and test/val data contain 1000 labeled images
2. Data Preprocessing
- We load the dataset and preprocess the images to prepare them for training. This includes resizing the images to a consistent size and normalizing the pixel values.We also split the dataset into training and validation sets to evaluate the model's performance.
3. Model Building
- We use a deep learning model, such as a Convolutional Neural Network (CNN), to learn features from the images and classify them into disease categories.The model consists of layers that extract patterns and features from the images, followed by fully connected layers for classification.¶

4. Training the Model:¶
- We train the model using the training dataset, adjusting the model's weights based on the difference between predicted and actual disease categories.During training, we monitor the model's performance on the validation set to avoid overfitting.
5. Model Evaluation:
- After training, we evaluate the model's performance on the validation set to assess its ability to generalize to new, unseen data.We use metrics such as accuracy to measure the model's performance.
6. Inference:
Once the model is trained and evaluated, we can use it to classify new tomato leaf images into disease categories.
