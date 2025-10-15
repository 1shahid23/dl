Experiment No: 1
Implement a multilayer perceptron (MLP) model using Keras with TensorFlow for house price prediction. (a)
Perform Exploratory Data Analysis (b) Prepare dataset (c) Build MLP model (d) Evaluate Model performance
(e) Predict for test data.
Topic Learning Objective:
Analyze the working of Artificial Neural Networks with forward and backpropagation.
Description:
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
data = pd.read_csv(&quot;./kc_house_data.csv&quot;)
data = data.drop(&#39;date&#39;,axis=1)
print(data)
data.corr()
price_corr = data.corr()[&#39;price&#39;].sort_values(ascending=False)
print(price_corr)
data.isnull().sum()
data = data.drop(&#39;id&#39;, axis=1)
data = data.drop(&#39;zipcode&#39;,axis=1)
x= data.drop(&#39;price&#39;,axis=1)
y= data[&#39;price&#39;]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
x_train = np.array(x_train).reshape(-1,17)
y_train = np.array(y_train).reshape(-1)
y_test = np.array(y_test).reshape(-1)
x_test = np.array(x_test).reshape(-1,17)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)

Lab Manual

Department of AIML SJEC
print(y_test.shape)
scaler = MinMaxScaler()
# fit and transfrom
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)
# everything has been scaled between 1 and 0
print(&#39;Max: &#39;,X_train.max())
print(&#39;Min: &#39;, X_train.min())
from keras.optimizers import SGD, Adam
model = Sequential()
model.add(layers.InputLayer((17,)))
model.add(layers.Dense(17,activation=&quot;relu&quot;))
model.add(layers.Dense(17,activation=&quot;relu&quot;))
model.add(layers.Dense(17,activation=&quot;relu&quot;))
model.add(layers.Dense(1,activation=&#39;linear&#39;))
o= Adam(learning_rate=0.1)
model.compile(loss=&#39;mean_squared_error&#39;,optimizer=o)
model.summary()
model.fit(X_train,y_train,epochs=500,batch_size=128,validation_data=(X_test,y_test))
y_pred = model.predict(X_test)
r2_score(y_test,y_pred)
sample_data = X_test[0:5, :].reshape(5, -1)
# Make a prediction
prediction = model.predict(sample_data)
print(&quot;\nActual Output:&quot;)
print(y_test[:5])
print(&quot;\nPredicted Output:&quot;)
print(prediction[:,0])

Experiment &amp; Result:

Lab Manual

Department of AIML SJEC
Experiment No: 2
Build a Multiclass classifier using Keras with TensorFlow. Use MNIST or any other suitable dataset. (a) Perform
Data Pre-processing (b) Define Model and perform training (c)Evaluate Results using confusion matrix.
Topic Learning Objective:
Analyze the working of an Artificial Neural Network for a classification task.
Description:
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from keras.models import Model
(training_image, training_labels),(test_image,test_labels) = mnist.load_data()
training_image= training_image/255.0
test_image=test_image/255.0
training_image=training_image.reshape(-1,28,28)
print(training_image.shape)
test_image=test_image.reshape(-1,28,28)
plt.imshow(training_image[58],cmap=&#39;gray&#39;)
plt.show()
plt.imshow(test_image[2],cmap=&#39;gray&#39;)
plt.show
from tensorflow.keras.utils import to_categorical
training_image=training_image.reshape(-1,28*28)
print(training_image.shape)
test_image=test_image.reshape(-1,28*28)
print(test_image.shape)
training_labels=to_categorical(training_labels,num_classes=10)
test_labels=to_categorical(test_labels,num_classes=10)
training_labels.shape
from keras.optimizers import Adam
from tensorflow.keras import Sequential
from keras import layers
model=Sequential()
model.add(layers.InputLayer((784,)))
model.add(layers.Dense(784,activation=&quot;relu&quot;))
model.add(layers.Dense(784,activation=&quot;relu&quot;))

Lab Manual

Department of AIML SJEC
model.add(layers.Dense(784,activation=&quot;relu&quot;))
model.add(layers.Dense(10,activation=&quot;softmax&quot;))
o=Adam(learning_rate=0.01)
model.compile(loss=&#39;categorical_crossentropy&#39;,optimizer=&quot;adam&quot;,metrics=[&#39;Accuracy&#39;])
model.summary()
model.fit(training_image,training_labels,validation_data=(test_image,test_labels),epochs=4)
sample_index = 2
sample_data = test_image[sample_index,:].reshape(1,-1)
prediction=model.predict(sample_data)
print(&quot;\nActual Output:&quot;)
print(np.argmax(test_labels[sample_index]))
print(&quot;Predicted Probabilities:&quot;)
print(prediction[0])
print(&quot;\n Predicted Output:&quot;)
print(np.argmax(prediction[0]))
y_pred = model.predict(test_image)
print(y_pred.shape)
print(test_labels.shape)
y_pred = np.argmax(y_pred, axis = -1)
y_test = np.argmax(test_labels, axis = -1)
print(y_pred.shape)
print(y_test.shape)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, \
recall_score, f1_score, classification_report
class_names = [&#39;0&#39;, &#39;1&#39;, &#39;2&#39;, &#39;3&#39;, &#39;4&#39;, &#39;5&#39;, &#39;6&#39;, &#39;7&#39;, &#39;8&#39;, &#39;9&#39;]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title(&quot;Confusion Matrix&quot;)
plt.show()
print(&quot;Accuracy:&quot;, accuracy_score(y_test, y_pred))
print(&quot;Precision (macro):&quot;, precision_score(y_test, y_pred, average=&#39;macro&#39;))
print(&quot;Recall (macro):&quot;, recall_score(y_test, y_pred, average=&#39;macro&#39;))
print(&quot;F1-score (macro):&quot;, f1_score(y_test, y_pred, average=&#39;macro&#39;))
# Detailed per-class metrics
print(&quot;\nClassification Report:&quot;)
print(classification_report(y_test, y_pred))
Experiment &amp; Result:

Lab Manual

Department of AIML SJEC
Experiment No: 3
Build a Multiclass classifier using Keras with TensorFlow. Use MNIST or any other suitable dataset. (a) Perform
Data Pre-processing (b) Define Model and perform training (c)Evaluate Results using confusion matrix.
Topic Learning Objective:
Analyze the working of an Convolutional Neural Network for a classification task.
Description:
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, ReLU, LeakyReLU, BatchNormalization, MaxPooling2D,
Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
plt.imshow(x_train[2])
print(&quot;label: &quot;,y_train[2])
print(x_train[2].shape)
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
IMG_SIZE = 48
def to_rgb_and_resize(x):
# x: (N, 28, 28) uint8
x = np.expand_dims(x, -1) # (N, 28, 28, 1)
x = np.repeat(x, 3, axis=-1) # (N, 28, 28, 3) RGB
x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE)).numpy() # float32 in [0,255]
return x
x_train = x_train / 255
x_test = x_test / 255
x_train = to_rgb_and_resize(x_train)
x_test = to_rgb_and_resize(x_test)
base = VGG16(
weights=None,
include_top=False,
input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.load_weights(&quot;vgg16_weights.h5&quot;)
for layer in base.layers[:-4]:
layer.trainable = False
model = Sequential([
layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
base, # you can put a Model inside Sequential

Lab Manual

Department of AIML SJEC
layers.GlobalAveragePooling2D(),
layers.Dense(128, activation=&#39;relu&#39;),
layers.Dropout(0.3),
layers.Dense(num_classes, activation=&#39;softmax&#39;),
])
model.compile(
optimizer=tf.keras.optimizers.SGD(1e-3),
loss=&#39;categorical_crossentropy&#39;,
metrics=[&#39;accuracy&#39;]
)
model.summary()
history = model.fit(
x_train, y_train,
validation_data=(x_test, y_test),
epochs=4,
batch_size=64,
shuffle=True,
verbose=1
)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f&quot;Test accuracy: {acc:.4f}&quot;)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis = -1)
y_test = np.argmax(y_test, axis = -1)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis = -1)
y_test = np.argmax(y_test, axis = -1)
Experiment &amp; Result:

Lab Manual

Department of AIML SJEC
Experiment No: 4
Design and implement a CNN model, ResNet-50, for Image Classification. (a) Define Model and perform training
(b) Evaluate Results using two performance measure metrics. Select a suitable image classification dataset.
Topic Learning Objective:
Analyze the working of the CNN model ResNet 50 for a classification task.
Description:
import os, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
# ✅ Hyperparameters
IMG_SIZE = 64 # ResNet-50 default
BATCH_SIZE = 64
EPOCHS = 20
SEED = 42
LR = 1e-3
NUM_CLASSES = 10
CLASS_NAMES = [&#39;airplane&#39;,&#39;automobile&#39;,&#39;bird&#39;,&#39;cat&#39;,&#39;deer&#39;,&#39;dog&#39;,&#39;frog&#39;,&#39;horse&#39;,&#39;ship&#39;,&#39;truck&#39;]
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Split off a validation set from train
x_train, x_val, y_train, y_val = train_test_split(
x_train, y_train, test_size=0.10, random_state=SEED, stratify=y_train
)
plt.imshow(x_train[2])
print(&quot;label: &quot;, CLASS_NAMES[y_train[2][0]])
print(x_train[2].shape)

y_train = to_categorical(y_train, NUM_CLASSES)

Lab Manual

Department of AIML SJEC
y_test = to_categorical(y_test, NUM_CLASSES)
def preprocess_and_resize(x):
x = x /255
x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE)).numpy() # float32 in [0,255]
return x

x_train = preprocess_and_resize(x_train)
x_test = preprocess_and_resize(x_test)
base = ResNet50(
weights=None,
include_top=False,
input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
for layer in base.layers[:-4]:
layer.trainable = False
base.load_weights(filepath= &quot;resnet50_weights.h5&quot;)
model = Sequential([
layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
base, # you can put a Model inside Sequential
layers.GlobalAveragePooling2D(),
layers.Dense(128, activation=&#39;relu&#39;),
layers.Dropout(0.3),
layers.Dense(NUM_CLASSES, activation=&#39;softmax&#39;),
])
model.compile(
optimizer=tf.keras.optimizers.Adam(LR),
loss=&#39;categorical_crossentropy&#39;,
metrics=[&#39;accuracy&#39;]
)
model.summary()
# %% Training
ckpt_path = &quot;resnet50_cifar10_best.h5&quot;
callbacks = [
ModelCheckpoint(ckpt_path, monitor=&quot;val_accuracy&quot;, save_best_only=True)
]

Lab Manual

Department of AIML SJEC
history = model.fit(
x_train, y_train,
validation_data=(x_test, y_test),
epochs=EPOCHS,
batch_size=64,
shuffle=True,
callbacks=callbacks,
verbose=1
)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f&quot;Test accuracy: {acc:.4f}&quot;)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis = -1)
y_test = np.argmax(y_test, axis = -1)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis = -1)
y_test = np.argmax(y_test, axis = -1)
import matplotlib.pyplot as plt
def plot_training(history):
# Extract metrics
acc = history.history[&#39;accuracy&#39;]
val_acc = history.history[&#39;val_accuracy&#39;]
loss = history.history[&#39;loss&#39;]
val_loss = history.history[&#39;val_loss&#39;]
epochs = range(1, len(acc) + 1)
# Plot Accuracy
plt.figure(figsize=(8, 5))
plt.plot(epochs, acc, &#39;bo-&#39;, label=&#39;Training Accuracy&#39;)
plt.plot(epochs, val_acc, &#39;ro-&#39;, label=&#39;Validation Accuracy&#39;)
plt.title(&#39;Training and Validation Accuracy&#39;)
plt.xlabel(&#39;Epochs&#39;)
plt.ylabel(&#39;Accuracy&#39;)
plt.legend()
plt.grid(True)
plt.show()
# Plot Loss

Lab Manual

Department of AIML SJEC
plt.figure(figsize=(8, 5))
plt.plot(epochs, loss, &#39;bo-&#39;, label=&#39;Training Loss&#39;)
plt.plot(epochs, val_loss, &#39;ro-&#39;, label=&#39;Validation Loss&#39;)
plt.title(&#39;Training and Validation Loss&#39;)
plt.xlabel(&#39;Epochs&#39;)
plt.ylabel(&#39;Loss&#39;)
plt.legend()
plt.grid(True)
plt.show()
# Call function
plot_training(history)

Experiment &amp; Result:

Lab Manual

Department of AIML SJEC
Experiment No: 5
Apply the transfer learning technique in deep neural network. Use the pre-trained model DenseNet on suitable
datasets.
Topic Learning Objective:
Apply transfer learning for training a CNN for the classification task.
Description:
import os, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
# ✅ Hyperparameters
IMG_SIZE = 64 # ResNet-50 default
BATCH_SIZE = 64
EPOCHS = 20
SEED = 42
LR = 1e-3
NUM_CLASSES = 10
CLASS_NAMES = [&#39;airplane&#39;,&#39;automobile&#39;,&#39;bird&#39;,&#39;cat&#39;,&#39;deer&#39;,&#39;dog&#39;,&#39;frog&#39;,&#39;horse&#39;,&#39;ship&#39;,&#39;truck&#39;]
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Split off a validation set from train
x_train, x_val, y_train, y_val = train_test_split(
x_train, y_train, test_size=0.10, random_state=SEED, stratify=y_train
)
plt.imshow(x_train[2])
print(&quot;label: &quot;, CLASS_NAMES[y_train[2][0]])

Lab Manual

Department of AIML SJEC
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)
def preprocess_and_resize(x):
x = x /255
x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE)).numpy() # float32 in [0,255]
return x

x_train = preprocess_and_resize(x_train)
x_test = preprocess_and_resize(x_test)
base = DenseNet121(
weights=None,
include_top=False,
input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
for layer in base.layers[:-4]:
layer.trainable = False
base.load_weights(filepath= &quot;DenseNet121_weights.h5&quot;)
base.summary()
model = Sequential([
layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
base, # you can put a Model inside Sequential
layers.GlobalAveragePooling2D(),
layers.Dense(128, activation=&#39;relu&#39;),
layers.Dropout(0.3),
layers.Dense(NUM_CLASSES, activation=&#39;softmax&#39;),
])
model.compile(
optimizer=tf.keras.optimizers.Adam(LR),
loss=&#39;categorical_crossentropy&#39;,
metrics=[&#39;accuracy&#39;]
)
model.summary()

Lab Manual

Department of AIML SJEC
# %% Training
ckpt_path = &quot;densenet121_cifar10_best.h5&quot;
callbacks = [
ModelCheckpoint(ckpt_path, monitor=&quot;val_accuracy&quot;, save_best_only=True)
]
history = model.fit(
x_train, y_train,
validation_data=(x_test, y_test),
epochs=EPOCHS,
batch_size=64,
shuffle=True,
callbacks=callbacks,
verbose=1
)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis = -1)
y_test = np.argmax(y_test, axis = -1)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, \
recall_score, f1_score, classification_report
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Method 1: Quick plot using sklearn&#39;s built-in display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap=plt.cm.Blues)
plt.title(&quot;Confusion Matrix&quot;)
plt.show()
print(&quot;Accuracy:&quot;, accuracy_score(y_test, y_pred))
print(&quot;Precision:&quot;, precision_score(y_test, y_pred, average=&#39;macro&#39;))
print(&quot;Recall:&quot;, recall_score(y_test, y_pred, average=&#39;macro&#39;))
print(&quot;F1-score:&quot;, f1_score(y_test, y_pred, average=&#39;macro&#39;))
# Detailed per-class metrics
print(&quot;\nClassification Report:&quot;)

Lab Manual

Department of AIML SJEC
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, \
recall_score, f1_score, classification_report
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap=plt.cm.Blues)
plt.title(&quot;Confusion Matrix&quot;)
plt.show()
print(&quot;Accuracy:&quot;, accuracy_score(y_test, y_pred))
print(&quot;Precision:&quot;, precision_score(y_test, y_pred, average=&#39;macro&#39;))
print(&quot;Recall:&quot;, recall_score(y_test, y_pred, average=&#39;macro&#39;))
print(&quot;F1-score:&quot;, f1_score(y_test, y_pred, average=&#39;macro&#39;))
# Detailed per-class metrics
print(&quot;\nClassification Report:&quot;)
print(classification_report(y_test, y_pred))
