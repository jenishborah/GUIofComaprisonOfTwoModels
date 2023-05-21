from flask import Flask, render_template, request
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


app = Flask(__name__)

# Define the load_dataset function before the compare function
def load_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset directory does not exist: " + dataset_path)

    # Load the dataset from the directory
    return dataset_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compare', methods=['POST'])
def compare():
    # Get the selected model names
    model1 = request.form.get('model1')
    model2 = request.form.get('model2')

    # Get the selected parameters
    epochs = int(request.form.get('epochs'))
    batch_size = int(request.form.get('batch_size'))

     # Get the form data
    dataset_link = request.form.get('dataset_link')
    dataset_path = request.form.get('dataset_path')

    if dataset_path is None:
        return "Please provide a valid dataset path."

    # Load the dataset
    dataset = load_dataset(dataset_path)

    # Preprocess the dataset based on the selected models
    preprocessed_data1, preprocessed_data2 = preprocess_data(dataset, model1, model2)


    # Train the models and generate the comparison graph
    history_model1, history_model2 = train_and_compare_models(model1, model2, epochs, batch_size, preprocessed_data1, preprocessed_data2)

    # Generate the comparison graph
    generate_comparison_graph(history_model1, history_model2)

    # Return the path to the generated comparison graph
    comparison_graph_path = os.path.join('static', 'comparison_graph.png')
    return render_template('result.html', comparison_graph_path=comparison_graph_path)
def preprocess_data(dataset, model1, model2):
    preprocessed_data1 = []
    preprocessed_data2 = []

    if model1 == 'jenish':
        # No additional preprocessing for your own model
        preprocessed_data1 = dataset
    elif model1 == 'efficientnet':
        # Implement the preprocessing steps for the EfficientNet model
        preprocessed_data1 = dataset
    elif model1 == 'alexnet':
        # Implement the preprocessing steps for the AlexNet model
        preprocessed_data1 = dataset

    if model2 == 'jenish':
        # No additional preprocessing for your own model
        preprocessed_data2 = dataset
    elif model2 == 'efficientnet':
        # Implement the preprocessing steps for the EfficientNet model
        preprocessed_data2 = dataset
    elif model2 == 'alexnet':
        # Implement the preprocessing steps for the AlexNet model
        preprocessed_data2 = dataset

    return preprocessed_data1, preprocessed_data2

def load_and_preprocess_dataset(dataset_path, batch_size, validation_split):
    # Generate data with augmentation for training and validation set
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split
    )

    # Load and preprocess the dataset
    data = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training' if validation_split < 1.0 else None
    )

    return data, None if validation_split == 1.0 else datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

from tensorflow.keras.models import load_model


def train_and_compare_models(model1, model2, epochs, batch_size, preprocessed_data1, preprocessed_data2):
    # Load and preprocess the dataset
    train_data1, val_data1 = load_and_preprocess_dataset(preprocessed_data1, batch_size, validation_split=0.2)
    train_data2, val_data2 = load_and_preprocess_dataset(preprocessed_data2, batch_size, validation_split=0.2)

    if model1 == 'jenish':
        # Define your own model architecture for model1 by fine-tuning NASNet Mobile
        base_model = tf.keras.applications.NASNetMobile(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(8, activation='softmax')(x)
        model1 = tf.keras.models.Model(inputs=base_model.input, outputs=x)

    elif model1 == 'efficientnet':
        if model2 == 'efficientnet':
    # Load or define the EfficientNet model architecture for model2
         base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers:
         layer.trainable = False

         x = base_model.output
         x = tf.keras.layers.GlobalAveragePooling2D()(x)
         x = tf.keras.layers.Dense(512, activation='relu')(x)
         x = tf.keras.layers.Dropout(0.5)(x)
        output = tf.keras.layers.Dense(8, activation='softmax')(x)

        model2 = tf.keras.models.Model(inputs=base_model.input, outputs=output)

    elif model1 == 'alexnet':
        # Load or define the AlexNet model architecture for model1
        model1 = tf.keras.applications.AlexNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    if model2 == 'jenish':
        # Define your own model architecture for model2 by fine-tuning NASNet Mobile
        base_model = tf.keras.applications.NASNetMobile(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        # Unfreeze the top layers of the base model
        for layer in base_model.layers[-120:]:
            layer.trainable = True
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(8, activation='softmax')(x)
        model2 = tf.keras.models.Model(inputs=base_model.input, outputs=x)

    elif model2 == 'efficientnet':
    # Load or define the EfficientNet model architecture for model2
        base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers:
         layer.trainable = False

         x = base_model.output
         x = tf.keras.layers.GlobalAveragePooling2D()(x)
         x = tf.keras.layers.Dense(512, activation='relu')(x)
         x = tf.keras.layers.Dropout(0.5)(x)
         output = tf.keras.layers.Dense(8, activation='softmax')(x)

         model2 = tf.keras.models.Model(inputs=base_model.input, outputs=output)
    
    elif model2 == 'alexnet':
        # Load or define the AlexNet model architecture for model2
        model2 = tf.keras.applications.AlexNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history_model1 = model1.fit(
        train_data1,
        epochs=epochs,
        validation_data=val_data1
    )

    model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history_model2 = model2.fit(
        train_data2,
        epochs=epochs,
        validation_data=val_data2
    )

    return history_model1, history_model2

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

def generate_comparison_graph(history_model1, history_model2):
    # Plot the training accuracy
    plt.plot(history_model1.history['accuracy'], label='Model 1 Training Accuracy')
    plt.plot(history_model2.history['accuracy'], label='Model 2 Training Accuracy')

    # Plot the validation accuracy
    plt.plot(history_model1.history['val_accuracy'], label='Model 1 Validation Accuracy')
    plt.plot(history_model2.history['val_accuracy'], label='Model 2 Validation Accuracy')

    # Set the x-axis label
    plt.xlabel('Epochs')

    # Set the y-axis label
    plt.ylabel('Accuracy')

    # Set the title of the graph
    plt.title('Comparison of Training and Validation Accuracy')

    # Add a legend to the graph
    plt.legend()

    # Save the comparison graph
    comparison_graph_path = os.path.join('static', 'comparison_graph.png')
    plt.savefig(comparison_graph_path)
    
    

if __name__ == '__main__':
    app.run(debug=True)
