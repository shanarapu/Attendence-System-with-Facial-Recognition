# Attendence-System-with-Facial-Recognition
The primary goal of this project is to design and implement a facial recognition system that is capable of accurately identifying individuals in real-time. 

**Data Collection Module**
The first phase of the system pipeline involves acquiring facial data from users, which is critical to the performance of subsequent modules. This process is performed using a real-time webcam feed and the OpenCV library. The purpose of this module is to capture clean, high-quality facial images under consistent lighting and positioning conditions to create a robust dataset.

• Live Camera Feed: When initiated, the webcam feed is activated and
displayed to the user in a window. The feed allows the user to position
themselves properly before image capture begins.

• Face Detection: The system uses face detection algorithms such as
Haar Cascade classifiers or deep neural network (DNN)-based detec-
tors provided by OpenCV. These algorithms identify facial regions in
each frame of the video feed in real time.

• Image Capture and Processing: Once a face is detected, it is cropped
from the frame. Preprocessing steps include resizing to a uniform size
(e.g., 112x92 pixels), grayscale conversion (if required by the model),
and optional histogram equalization to enhance contrast.

• Data Storage Structure: The cropped and preprocessed images are
saved in a structured directory format. Each user has a uniquely la-
beled folder (using their name, ID, or label index), and images are saved
sequentially within that folder.

• Configurable Parameters: Users can specify the number of images to
be collected for each person (for example, 150-200 images). The system
automatically counts and stops when the required number is reached,
avoiding redundant storage.

• User Feedback: Visual feedback such as progress bars or frame coun-
ters is provided to guide the user during data collection. In cases of low
lighting or unclear faces, warnings are displayed to prompt corrections.

This module ensures the consistency and quality of facial images, which
is critical for training an accurate and generalizable Convolutional Neural
Network (CNN) model in the later stages.

**Dataset Creation and Preprocessing**

Once the facial images have been collected and organized, the next step in-
volves preparing the data for training the CNN model. This includes labeling,
splitting, and preprocessing the images to ensure compatibility with the deep
learning pipeline.

• Directory-Based Labeling: The collected dataset is organized in a
directory structure where each folder represents a unique user ID or
label. This structure allows for automatic label extraction during train-
ing using image data generators or custom scripts.

• Training and Validation Split: To evaluate model generalization,
the dataset is divided into training and validation sets. A common ra-
tio (e.g., 80:20) is used, with the majority of the images going into the
training set and the rest reserved for validation during training.

• Image Resizing: All images are resized to a uniform dimension (e.g.,
112x92 pixels) to standardize input size for the CNN. This also helps re-
duce computational overhead without sacrificing important facial fea-
tures.

• Color Channel Adjustment: Depending on the model configuration,
images are either converted to grayscale (for simpler models) or kept in
RGB format. Grayscale conversion reduces complexity but may impact
accuracy if color-based features are important.

• Normalization: Pixel values are normalized to a [0,1] range by di-
viding each pixel value by 255. This ensures faster and more stable
convergence during model training by preventing gradient explosions
or vanishing issues.

These preprocessing steps ensure that the dataset is clean, balanced, and
ready for training. By enforcing consistency in image size, format, and nor-
malization, this module plays a critical role in achieving high model perfor-
mance and reliability.

**Model Training**

Once the dataset is preprocessed and ready, the next step involves training
a Convolutional Neural Network (CNN) to perform facial recognition. The
model training process is conducted using TensorFlow and Keras, which offer
a high-level interface for building and training deep learning models.

• Model Architecture: The CNN model is composed of several layers:
– Input Layer: Accepts the preprocessed image input of fixed dimensions.
– Convolutional Layers: These layers extract spatial features from
the input image using multiple filters. ReLU (Rectified Linear
Unit) activation is applied to introduce non-linearity.

– Pooling Layers: MaxPooling layers are used to reduce dimensionality and control overfitting by downsampling feature maps.

– Dropout Layers: Dropout is applied to randomly deactivate neurons during training, reducing overfitting and improving general-
ization.

– Fully Connected (Dense) Layers: These layers interpret the
learned features and help classify the image into one of the user
IDs.

– Output Layer: A softmax activation function is used in the out-
put layer for multi-class classification, where each class corre-
sponds to a registered user.

• **Compilation**: The model is compiled using:

– categorical_crossentropy loss function for multi-class classification.
– Adam optimizer (adaptive learning rate) for faster convergence.
– accuracy as a performance metric to monitor training progress.

• Training Process: The training is executed over a set number of
epochs (100 epochs), with mini-batch processing (batch size of 16 or 32).
A validation set is used to monitor overfitting and guide early stopping
or model checkpointing.

• Evaluation: Post-training, the model is evaluated on the validation
dataset to assess generalization performance. Metrics such as accuracy,
loss, precision, and recall are calculated and plotted for analysis.

• Model Saving: Upon satisfactory performance, the trained model is
saved in HDF5 format (e.g., model.h5) for future use. This file contains
both the model structure and the learned weights.

• Training Logs and Visualization: During training, logs of loss and
accuracy for both training and validation sets are recorded. These met-
rics are visualized using matplotlib to provide insights into learning
behavior and detect issues like underfitting or overfitting.

This module is crucial to the system’s accuracy and reliability. A well-
trained CNN model ensures high confidence in real-time recognition scenar-
ios and contributes to a seamless user experience.

**Face Recognition and Inference**

Once the Convolutional Neural Network (CNN) model has been successfully
trained and saved, it is deployed in the real-time recognition pipeline. This
module is responsible for detecting faces from a live camera feed, prepro-
cessing them similarly to the training phase, and classifying them using the
trained model.

• Model Loading: The trained model is loaded from the saved .h5 file
using Keras’ load_model() function. This enables the system to use
the trained weights for inference without retraining.

• Live Frame Capture: The system continuously accesses frames from
a webcam using OpenCV. Each frame is processed in real-time to detect
any faces present using a Haar Cascade face detector.

• Face Preprocessing: Once a face is detected in the frame:

– It is cropped and resized to match the input dimensions of the
CNN model.
– Pixel values are normalized and optionally converted to grayscale
images.
– The processed face image is reshaped to match the model’s expected input shape and passed to the CNN for prediction.

• Prediction and Output: The CNN model predicts the identity of the
person in the frame:
– The softmax output layer returns a probability distribution over
all known classes (users).
– The class with the highest probability is selected as the predicted
identity.
– A confidence score is calculated based on the probability of the top
prediction.

• **Result Display**: The recognized identity (name or ID) and confidence
score are overlaid onto the video feed in real time. This gives immediate
feedback to users and helps verify recognition success.

• Performance Optimization: To maintain smooth performance, the
system may employ:
– Frame Skipping: Only every nth frame is processed to reduce
CPU/GPU load.
– Threading: Face detection and recognition operations can be run
in separate threads to prevent frame drops and improve responsiveness.

• Multi-face Handling: The system supports detection of multiple faces
in a single frame. Each detected face is processed individually, enabling
recognition of more than one person simultaneously if needed.
• Error Handling: If no face is detected or if the model fails to classify
the input with sufficient confidence, a message such as “Unknown” is
displayed, and the event is logged for analysis.
This module acts as the real-time interface of the system, providing immediate and intuitive recognition results. Its robustness and accuracy are
directly influenced by the quality of training data and preprocessing techniques.

**Logging and Reporting**
The Logging and Reporting module is responsible for maintaining a persistent record of recognition events. Each time a face is successfully recognized
by the system, a detailed log is created and stored locally. These logs serve as
the foundation for generating meaningful reports and summaries of system
usage.

• Structured Logging:
– Each recognition event is recorded in a structured format using
JSON.
– The log entry includes relevant fields such as:
* User ID: A unique identifier for the recognized individual. In
our case "Roll.no"
* Name: The user’s name associated with the prediction class.
* Branch: The user’s branch is also logged.
* Date and Time: Timestamp of the recognition event.
• Local Storage:
– Logs are saved to disk in a time-indexed JSON file, ensuring easy
access and filtering.
– The modular nature of the logging mechanism makes it adaptable
to other storage formats such as CSV, SQLite, or even cloud-based
databases if needed in the future.
• Reporting Features:
– Logs can be parsed to generate:
* Attendance Reports: Useful for applications like classrooms
or offices.
* Usage Analytics: Number of recognitions per user, peak recog-
nition times, system uptime.
* Error Trends: Analyze logs for unrecognized attempts or
low-confidence matches.
– Reports can be exported in various formats (e.g., CSV, Excel, PDF)
for administrative or record-keeping purposes.
38
