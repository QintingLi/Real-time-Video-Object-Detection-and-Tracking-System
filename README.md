Pedestrian and Vehicle Tracking using OpenCV

This project uses OpenCV and custom algorithms to detect and track pedestrians and vehicles in a video feed. 
The script is designed to handle two modes of operation: background subtraction-based detection and DNN-based object detection.

Requirements

To run the script, you need to have the following libraries installed:

Python 3.x
OpenCV (cv2)
NumPy

You can install the required libraries using pip:
pip install numpy opencv-python


Usage

The script can be run from the command line with different modes depending on the desired task. It takes two arguments:

The mode flag (-b for background subtraction or -d for DNN-based detection).
The path to the video file to process.
Command Line Arguments
-b: Run the background subtraction-based detection.
-d: Run the DNN-based object detection.
Example Commands

To run the background subtraction mode:
python tracking_script.py -b path/to/video.mp4


Detailed Description of the Code
Resizing Frames: The function resize_vga(frame) resizes input frames to a resolution of 640x480 for consistent processing.
Connected Components Algorithm: my_connected_components(data) is a custom implementation based on the Breadth-First Search (BFS) algorithm to find connected components in a binary image. It labels connected regions (representing detected objects) and returns their bounding boxes.
Classifier Function: classifier(cnt, record) classifies detected objects based on their sizes into humans, cars, and others.
Background Estimation: estimated_background(cap) estimates a background image from the video using a random sample of frames.
Box Similarity: similar_box(pos1, pos2, delta) determines whether two bounding boxes are similar based on Manhattan distance.
Main Execution Flow: The script reads the command-line arguments to determine which detection mode to use and processes the video accordingly.
Background Subtraction Mode (-b): This mode uses a Gaussian Mixture Model (GMM) to subtract the background and identify moving objects. Detected objects are processed through morphological operations and classified using connected components.
DNN-based Detection Mode (-d): This mode uses a pre-trained SSD MobileNet model for detecting objects (specifically focusing on people). It tracks moving objects across frames using a simple Manhattan distance-based tracking mechanism.
Visualization: The results of both modes are visualized in a window displaying the original frame, detected objects, and their classifications. Press Esc to exit the visualization window.
DNN Model Files
If you choose to run the DNN-based detection mode (-d), make sure to have the following model files in the same directory as the script:

frozen_inference_graph.pb
ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt
object_detection_classes_coco.txt
These files are required to run the DNN-based object detection using TensorFlow's pre-trained models.

Exit the Program
To safely exit the program while it is running, press the Esc key.

Error Handling
The script includes basic error handling for file opening errors and runtime errors.

Author

This script was created as a demonstration of pedestrian and vehicle tracking using computer vision techniques.

License

This project is licensed under the MIT License - see the LICENSE file for details.
