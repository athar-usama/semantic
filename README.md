# semantic
Semantic segmentation on Cityscapes Dataset with U-Net as the baseline network and MobileNet &amp; ResNeXt as backbone encoders.

## Contents
1. [ Abstract ](#abs)
2. [ Methodology Diagram ](#m_dig)
3. [ Introduction ](#intro)
4. [ Setup Instructions ](#setup)
5. [ Usage Instructions ](#usage)
6. [ Quantitative Results ](#quant_res)
7. [ Training Graphs ](#graphs)
8. [ Visual Results ](#vis_res)
9. [ Author ](#auth)

<a name="abs"></a>
## Abstract
This report presents the results of semantic segmentation notebook, focusing on the classification of pixels in an image into predefined classes. The assignment involves training and evaluating two convolutional neural network (CNN) architectures as encoders for a segmentation network on the provided dataset. The report highlights the importance of semantic segmentation in scene understanding and explains the process of dataset preparation. It recommends the use of data augmentation and transfer learning techniques, emphasizing the importance of well-documented and modular code. It contains network details, training graphs, performance measures, and illustrative images showcasing the original image, ground truth segmentation masks, and predicted masks. This report aims to provide a comprehensive overview of the performed work as well as its outcomes.

<a name="m_dig"></a>
## Methodology Diagram
![methodology_diagram](https://github.com/athar-usama/semantic/assets/41828100/e68b2ac8-9533-4e78-a74c-405293305bb4)

<a name="intro"></a>
## Introduction
Semantic image segmentation is a fundamental task in the field of computer vision, aiming to classify each pixel in an image into predefined classes. It plays a crucial role in scene understanding and has numerous applications, including autonomous driving, object recognition, image editing, and augmented reality. By providing a pixel-level understanding of an image, semantic segmentation enables machines to perceive and interpret visual data more effectively. The objective of this report is to present the findings and results of our assignment on semantic segmentation. It focuses on training and evaluating two different convolutional neural network (CNN) architectures
as backbones for a semantic segmentation network on the Cityscapes Dataset.

Semantic segmentation differs from object detection, as it does not involve predicting bounding boxes or distinguishing between instances of the same object. Instead, it aims to assign a semantic label to each pixel in the image, providing a comprehensive understanding of the scene. In an outdoor image, semantic segmentation can help differentiate between sky, ground, trees, people, and other objects or entities present.

To perform semantic segmentation effectively, a higher-level understanding of the image is required. The algorithm needs to identify objects present in the scene and accurately assign the corresponding pixels to each object. Dataset preparation plays a crucial role in training a segmentation model. This involves obtaining input RGB images and creating corresponding segmentation images with pixel-level class labels.

We were provided with a subset derived from the Cityscapes Dataset, which is already divided into training and testing splits. The dataset encompasses a range of classes, including sky, building, pole, road, pavement, tree, sign symbol, fence, car, pedestrian, and bicyclist. The goal is to train a semantic segmentation network with two different encoders and compare their qualitative and quantitative performance.

We employed U-Net for this and selected ResNeXt and MobileNet as the 2 encoders. To achieve robust segmentation results, we employed data augmentation and transfer learning techniques. Data augmentation helps in increasing the size and diversity of the training dataset by applying transformations like rotations, scaling, and flipping. Transfer learning leverages pretrained models on large-scale datasets to initialize our own models, enabling them to learn more efficiently with a smaller dataset.

The deliverables for the assignment include the implementation of the task using Google Colab. The code files are uploaded to a GitHub repository, and a link to the repository is included at the end of this report. The Python Notebook demonstrates a neatly written, documented, and modular code. It also includes quantitative performance measures (F1 Score, Dice coefficient, accuracy, IoU, sensitivity, and specificity) as well as qualitative results showcasing correctly and incorrectly classified images in the requested format.

<a name="setup"></a>
## Setup Instructions
Following are the instructions for setting up an environment for running this code.

The code performs semantic segmentation on the subset of Cityscapes Dataset and compares the implementation of two different backbone encoders (MobileNet & ResNeXt) for the basline segmentation network (U-Net). The pretrained baselines used in this project were imported from PyTorch models.

### Requirements:
To run this code, you will need the following:

1- Python</br>
2- OpenCV</br>
3- PyTorch</br>
4- TensorFlow</br>
5- Pandas</br>
6- scikit-learn

There are 2 methods for running this code. Let us take a brief look at each one of them.

### Method 1 (On Cloud):

Download the Python notebook and open it up inside Google Colaboratory. The datasets will be automatically mound via Google Drive.

Just run the notebook and wait for the models to train themselves. Make sure to run only that model's block which you intend to train. These models take some minutes to go through all 20 epochs on the Cityscapes Dataset. During training, each epoch is saved in checkpoints and the best performing model is also dumped in Google Drive at the end.

### Method 2 (On Device):

Clone this repository to your local machine with the following command:</br>
<pre>git clone https://github.com/athar-usama/semantic.git</pre>

Install Python and Jupyter Notebook. You can download them from the official websites:
#### Python: https://www.python.org/downloads/
#### Jupyter Notebook: https://jupyter.org/install

Install OpenCV, PyTorch, TensorFlow, Pandas, and scikit-learn libraries using pip:</br>
<pre>pip install opencv-python pytorch tensorflow pandas scikit-learn</pre>
<pre>pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118</pre>

Launch Jupyter Notebook from the command line:</br>
<pre>jupyter notebook</pre>

Open the .ipynb file in Jupyter Notebook and run the code.

<a name="usage"></a>
## Usage Instructions
The code is designed to work with a subset of the Cityscapes Dataset. Both of these datasets have already been uploaded on a publicly accessible Google Drive folder. That folder gets downloaded from inside the Google Colaboratory notebook. Therefore, there is no need to download it locally.

All you need is to run the code and wait for the models to be trained.

<a name="quant_res"></a>
## Quantitative Results
![quant](https://github.com/athar-usama/semantic/assets/41828100/ff60187c-af65-45ac-a654-56f004f445eb)

<a name="graphs"></a>
## Training Graphs

### With ResNeXt Ecnoder

#### Accuracy Curves
![acc_res](https://github.com/athar-usama/semantic/assets/41828100/c6050257-fd48-4386-96e6-95f7d910bcff)

#### Loss Curves
![loss_res](https://github.com/athar-usama/semantic/assets/41828100/5168c28a-4d16-4e8d-8cf8-b8c0cc2b2da4)

#### IoU Curves
![iou_res](https://github.com/athar-usama/semantic/assets/41828100/1c567843-00c8-404d-a4a9-93047180edd5)

### With MobileNet Ecnoder

#### Accuracy Curves
![acc_mob](https://github.com/athar-usama/semantic/assets/41828100/8264cb1d-13f9-40bb-86dc-d807a4962d98)

#### Loss Curves
![loss_mob](https://github.com/athar-usama/semantic/assets/41828100/d800a97c-8129-41e4-bd8c-90c125a01222)

#### IoU Curves
![iou_mob](https://github.com/athar-usama/semantic/assets/41828100/2a61bd5b-ac25-4b11-994b-0402568a6a9a)

<a name="vis_res"></a>
## Visual Results

### With ResNeXt Encoder
![qual_res](https://github.com/athar-usama/semantic/assets/41828100/1e09749d-87df-43a0-83c0-7879d452f213)

### With MobileNet Encoder
![qual_mob](https://github.com/athar-usama/semantic/assets/41828100/dd317356-32ac-4ee3-ba8a-e6d463c9bfb7)

<a name="auth"></a>
## Author
Usama Athar atharusama99@gmail.com
