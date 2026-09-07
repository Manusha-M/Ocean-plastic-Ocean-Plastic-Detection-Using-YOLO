# Ocean Plastic Detection Using YOLO

## Overview

Ocean plastic pollution is a major environmental problem. Detecting plastic waste manually from large amounts of ocean images and videos can be time-consuming.

This project uses **YOLOv4 (You Only Look Once)**, a deep learning-based object detection algorithm, to detect plastic waste in ocean images and video frames.

The project also includes image enhancement techniques to improve visibility in underwater and low-quality images before performing object detection.

## Objectives

* Detect plastic waste in ocean images.
* Detect plastic waste from video frames.
* Improve detection under challenging underwater conditions.
* Apply image enhancement techniques to improve image quality.
* Display detected objects using bounding boxes and confidence scores.
* Provide a foundation for automated marine pollution monitoring.

## Technologies Used

| Technology         | Purpose                           |
| ------------------ | --------------------------------- |
| Python             | Application development           |
| YOLOv4             | Object detection                  |
| PyTorch            | Deep learning and model inference |
| OpenCV             | Image and video processing        |
| NumPy              | Numerical operations              |
| Dark Channel Prior | Image enhancement and dehazing    |
| Shell Script       | Video detection workflow          |

## System Workflow

```text
Input Image / Video
        |
        v
Image Preprocessing
        |
        v
Image Enhancement / Dehazing
        |
        v
YOLOv4 Object Detection
        |
        v
Bounding Boxes and Confidence Scores
        |
        v
Rule-Based Classification
        |
        v
Final Detection Result
```

## Project Structure

```text
Ocean-plastic-Ocean-Plastic-Detection-Using-YOLO/
│
├── 60_epochs_denoised.pt
├── README.md
│
├── app.py
├── app2.py
├── main_app.py
│
├── inference.py
├── dark_channel_prior.py
├── rule_based_classifier.py
│
├── obj.data
├── ocean_plsatic.png
│
└── video_yolov4.sh
```

## File Description

| File                       | Description                                       |
| -------------------------- | ------------------------------------------------- |
| `60_epochs_denoised.pt`    | Trained YOLO model weights                        |
| `inference.py`             | Performs object detection using the trained model |
| `dark_channel_prior.py`    | Performs image enhancement/dehazing               |
| `rule_based_classifier.py` | Contains rule-based classification logic          |
| `app.py`                   | Application implementation                        |
| `app2.py`                  | Alternative application implementation            |
| `main_app.py`              | Main application entry point                      |
| `video_yolov4.sh`          | Script for video-based detection                  |
| `obj.data`                 | Object detection configuration                    |
| `ocean_plsatic.png`        | Project image/sample input                        |

## Model

The project uses a trained YOLOv4 model stored in:

```text
60_epochs_denoised.pt
```

The model is used during inference to identify plastic waste from input images.

The general detection process is:

```text
Input Image
    |
    v
Preprocessing
    |
    v
YOLOv4 Model
    |
    v
Object Detection
    |
    v
Bounding Box + Confidence Score
```

## Image Enhancement

Underwater and ocean images can contain several challenges, including:

* Low visibility
* Haze
* Poor contrast
* Color distortion
* Uneven lighting
* Image noise

The project includes a **Dark Channel Prior** implementation to improve image quality and visibility.

The enhanced image can then be provided to the object detection pipeline.

```text
Original Image
      |
      v
Dark Channel Prior
      |
      v
Enhanced Image
      |
      v
YOLOv4 Detection
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/Manusha-M/Ocean-plastic-Ocean-Plastic-Detection-Using-YOLO.git
```

### Navigate to the Project

```bash
cd Ocean-plastic-Ocean-Plastic-Detection-Using-YOLO
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install numpy opencv-python torch torchvision
```

Additional dependencies may be required depending on the application file being executed.

## Running the Project

### Image Detection

Run the inference script:

```bash
python inference.py
```

The script loads the trained model and performs object detection on the input image.

### Application

The application can be started using:

```bash
python app.py
```

or:

```bash
python main_app.py
```

### Video Detection

The repository contains a shell script for video-based detection:

```bash
bash video_yolov4.sh
```

## Input and Output

### Input

The system can process:

* Ocean images
* Underwater images
* Video frames

### Output

The detection system provides:

* Detected plastic objects
* Bounding boxes
* Confidence scores
* Classification results

## Example Use Case

A camera mounted on a boat, underwater vehicle, or marine monitoring system can capture images or video.

The system can process the captured data through the following pipeline:

```text
Camera
  |
  v
Ocean Image / Video
  |
  v
Image Enhancement
  |
  v
YOLOv4
  |
  v
Plastic Detection
  |
  v
Marine Pollution Monitoring
```

This type of system can potentially support:

* Marine pollution monitoring
* Ocean cleanup operations
* Environmental research
* Marine conservation
* Automated waste detection

## Advantages

* Automated plastic waste detection
* Supports image and video input
* Uses deep learning-based object detection
* Includes image enhancement for underwater conditions
* Can detect multiple objects in a scene
* Can be extended to additional marine waste categories

## Limitations

The detection performance may decrease when:

* Plastic objects are very small.
* Objects are partially hidden.
* Water visibility is extremely poor.
* Lighting conditions vary significantly.
* Plastic objects have similar visual characteristics to the surrounding environment.
* Input images differ significantly from the training data.

## Evaluation

The following metrics can be used to evaluate the object detection model:

| Metric    | Description                                                      |
| --------- | ---------------------------------------------------------------- |
| Precision | Measures the proportion of correct positive predictions          |
| Recall    | Measures how many actual objects were detected                   |
| IoU       | Measures the overlap between predicted and actual bounding boxes |
| mAP       | Measures overall object detection performance                    |
| F1-Score  | Combines precision and recall                                    |

Actual metric values should be added here after evaluating the trained model.

## Future Improvements

* Train the model using a larger and more diverse dataset.
* Add additional categories of marine waste.
* Compare YOLOv4 with newer YOLO architectures.
* Improve underwater image enhancement.
* Optimize the model for real-time inference.
* Add object tracking for video detection.
* Deploy the model on edge devices.
* Develop a web-based monitoring dashboard.
* Add detection statistics and pollution analytics.
* Improve model evaluation using precision, recall, mAP, and F1-score.

## Project Information

| Category             | Details                                   |
| -------------------- | ----------------------------------------- |
| Project Name         | Ocean Plastic Detection Using YOLOv4      |
| Domain               | Artificial Intelligence / Computer Vision |
| Programming Language | Python                                    |
| Detection Algorithm  | YOLOv4                                    |
| Input                | Images and Videos                         |
| Output               | Plastic Detection with Bounding Boxes     |
| Image Processing     | Dark Channel Prior                        |
| Model                | `60_epochs_denoised.pt`                   |

## Author

**Manusha Manne**

GitHub: [Manusha-M](https://github.com/Manusha-M)

## License

This project is intended for educational and research purposes.
