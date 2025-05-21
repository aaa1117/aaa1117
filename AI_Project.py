#packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(123)

import tensorflow as tf
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Create necessary directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'archive', 'HAM10000_images_part_1')
HISTORY_DIR = os.path.join(PROJECT_ROOT, 'history')

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, HISTORY_DIR]:
    os.makedirs(directory, exist_ok=True)

def plot_model_history(model_history):


    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # Summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1, len(model_history.history['accuracy'])//10))
    axs[0].legend(['train', 'val'], loc='best')
    # Summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1, len(model_history.history['loss'])//10))
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

# Function to check permissions
def check_permissions(directory):
    print(f"Checking permissions for: {directory}")
    print("Readable:", os.access(directory, os.R_OK))
    print("Writable:", os.access(directory, os.W_OK))
    print("Executable:", os.access(directory, os.X_OK))

# Set the correct path to your images directory
base_skin_dir = IMAGES_DIR

# Check permissions
check_permissions(base_skin_dir)

# Print the files found
files = glob(os.path.join(base_skin_dir, "*.jpg"))
#print("Files found:", files)

# Create a dictionary with image IDs and their paths
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in files}

# Dictionary for lesion types
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

# Verify the path
print(base_skin_dir)

#reading data and processe it 
skin_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'HAM10000_metadata.csv'))

# Creating New Columns for better readability

skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get)
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes

skin_df.head()

#data cleaning
skin_df.isnull().sum()
skin_df['age'].fillna((skin_df['age'].mean()), inplace=True)
skin_df.isnull().sum()
print(skin_df.dtypes)
duplicate_rows_count = len(skin_df[skin_df.duplicated(['image_id'])])
print("Number of duplicate data:", duplicate_rows_count)

fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
skin_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)
skin_df['dx_type'].value_counts().plot(kind='bar')
skin_df['localization'].value_counts().plot(kind='bar')
skin_df['age'].hist(bins=40)
skin_df['sex'].value_counts().plot(kind='bar')



#loading the images
skin_df = skin_df[skin_df['path'].notnull()]
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((224,224))) if os.path.exists(x) else None)

n_samples = 5
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs,
                                         skin_df.sort_values(['cell_type']).groupby('cell_type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
fig.savefig('category_samples.png', dpi=300)
skin_df['image'].map(lambda x: x.shape).value_counts()
features=skin_df.drop(columns=['cell_type_idx'],axis=1)
target=skin_df['cell_type_idx']

# Create the specified directory if it doesn't already exist
save_path = os.path.join(PROJECT_ROOT, 'sample_images')
os.makedirs(save_path, exist_ok=True)

for type_name, type_rows in skin_df.sort_values(['cell_type']).groupby('cell_type'):
    
    sample_row = type_rows.sample(1, random_state=1234).iloc[0]
    img_array = sample_row['image']
    
    img = Image.fromarray(img_array)

    
    img_file_path = os.path.join(save_path, f"{type_name}.jpg")
    img.save(img_file_path)

    
    plt.figure(figsize=(4, 3))
    plt.imshow(img_array)  
    plt.title(type_name)
    plt.axis('off')
    plt.show()

print(f"Images saved in the directory: {save_path}")

#test 
x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.20,random_state=1234,stratify=target)

#normalization
x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)

x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std

#label encoding
y_train = to_categorical(y_train_o, num_classes = 7)
y_test = to_categorical(y_test_o, num_classes = 7)

x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.1, random_state=42, stratify=y_train_o)
x_train = x_train.reshape(x_train.shape[0], *(224, 224, 3))
x_test = x_test.reshape(x_test.shape[0], *(224, 224, 3))
x_validate = x_validate.reshape(x_validate.shape[0], *(224, 224, 3))

#cnn
from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,AveragePooling2D,GlobalMaxPooling2D

input_shape = (224, 224, 3)
num_classes = 7

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Activation('relu'))
model.add(Dropout(0.25))

#Output
model.add(BatchNormalization())
model.add(Dense(num_classes, activation='softmax'))
model.summary()

#setting the optimizer
from tensorflow.keras.optimizers import Adam
# Define the optimizer
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

model_dir = MODEL_DIR
os.makedirs(model_dir, exist_ok=True)  


model_checkpoint = ModelCheckpoint(
    os.path.join(model_dir, 'best_model_Cnn_224x224.keras'),
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=4,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

early_stopping_monitor = EarlyStopping(patience=10,monitor='val_accuracy')

datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False, 
        rotation_range=10,  
        zoom_range = 0.1,
        width_shift_range=0.1, 
        height_shift_range=0.1,  
        horizontal_flip=False,  
        vertical_flip=False)  

datagen.fit(x_train)


import json
import scipy.io
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

# Fit the model
epochs = 50
batch_size = 20
history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                    epochs=epochs, validation_data=(x_validate, y_validate),
                    verbose=1, steps_per_epoch=x_train.shape[0] // batch_size,
                    callbacks=[learning_rate_reduction, model_checkpoint, early_stopping_monitor])

# Convert history to a dictionary
history_dict = history.history

# Define the save path for history
save_path = HISTORY_DIR
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Save the history as a .mat file
scipy.io.savemat(os.path.join(save_path, 'cnn_history.mat'), history_dict)

# Save the history as a CSV file
df_history = pd.DataFrame(history_dict)
df_history.to_csv(os.path.join(save_path, 'cnn_history.csv'), index=False)

# Plot and save the history as an image
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history_dict['accuracy'], label='Train Accuracy')
plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history_dict['loss'], label='Train Loss')
plt.plot(history_dict['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(save_path, 'cnn_history.png'))

# Make predictions on the test set
test_predictions = model.predict(x_test)
test_predictions_classes = np.argmax(test_predictions, axis=1)
test_true_classes = np.argmax(y_test, axis=1)

# Save test set predictions and true results to CSV
df_test_results = pd.DataFrame({'True_Labels': test_true_classes, 'Predicted_Labels': test_predictions_classes})
df_test_results.to_csv(os.path.join(save_path, 'cnn_test_results.csv'), index=False)

# Save test set predictions and true results to .mat
test_results_dict = {
    'True_Labels': test_true_classes,
    'Predicted_Labels': test_predictions_classes
}
scipy.io.savemat(os.path.join(save_path, 'cnn_test_results.mat'), test_results_dict)

print(f"Training history saved as 'cnn_history.mat', 'cnn_history.csv', and 'cnn_history.png' in {save_path}.")
print(f"Test set predictions vs true results saved as 'cnn_test_results.csv' and 'cnn_test_results.mat' in {save_path}.")
