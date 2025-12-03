# Powerlifting Athlete Pose Estimation and Lift Phase Analysis using YOLOv8-Pose

A pose-based computer vision system for analyzing Olympic weightlifting videos and automatically classifying the 14 distinct phases of the clean and jerk lift using YOLOv8 pose estimation.


https://github.com/user-attachments/assets/6abc05e4-e870-4d15-a3c6-c18abe910b7a


## Overview

This project implements an automated system for detecting and classifying powerlifting movements, specifically the "clean and jerk" Olympic lift. By leveraging YOLOv8-Pose for keypoint detection, the system can identify 14 distinct lift phases in real-time, providing valuable feedback for athletes and coaches.

## Features

- **Automated Phase Detection**: Identifies 14 distinct phases of the clean and jerk lift
- **Real-time Analysis**: Processes videos frame-by-frame with pose estimation overlays
- **Interactive Interface**: Streamlit-based web application for easy video upload and analysis
- **Custom Training**: Model trained on 679 manually annotated frames with 16 keypoints per frame
- **Visual Feedback**: Displays bounding boxes, keypoints, and detected phase sequences

## Lift Phases Detected

1. First Pull
2. Second Pull
3. Turn Over
4. Front Rack Catch
5. Front Rack Recovery
6. Prepare for Dip
7. Drive
8. Drop Under
9. Overhead Catch
10. Overhead Catch Recovery
11. Stabilize Weight Overhead
12. Drop Weight
13. Dip
14. Prepare for Lift

## Repository Structure
repo/
├── app.py # Streamlit web application
├── powerlifting-pose.pt # Trained YOLOv8 pose model
├── src/ # Source code and training notebooks
├── data/ # Dataset and annotations
├── runs/ # Training run results
├── runs2/ # Additional training experiments
├── runs3/ # Additional training experiments
└── sample_videos/ # Sample videos for testing
## Model Performance

<img width="4000" height="1200" alt="results" src="https://github.com/user-attachments/assets/600f3841-cd7a-4518-b352-aa37b8fbc0c0" />
<img width="3000" height="2250" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/6ff02272-c187-4b2c-85a2-2d1d4e211df3" />
<img width="2250" height="1500" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/d1badec4-af12-40bb-864c-0203f2b08222" />
<img width="2250" height="1500" alt="PoseF1_curve" src="https://github.com/user-attachments/assets/3e322570-1a31-427a-a546-25403de9291d" />

### Training Results
- **Box Detection**: Training precision/recall of 0.6-0.8, mAP50(B) up to ~0.8
- **Pose Detection**: Training mAP50(P) of 0.35-0.4
- **Training Time**: 1 hour 12 minutes on Google Colab GPU

### Validation Results
- **Box Detection**: Validation precision/recall around 0.5
- **Pose Detection**: Validation mAP50(P) of 0.25-0.3
- **Peak F1 Scores**: Box F1 of 0.5 at confidence 0.19, Pose F1 of 0.27 at confidence 0.84

### Performance Notes
- Strong performance on temporally distinct phases (First Pull, Second Pull, Prepare for Lift)
- Challenges with short transition phases (Drive, Overhead Catch Recovery, Drop Weight)
- Pose localization identified as the primary bottleneck
- No signs of overfitting observed

## Dataset

- **Total Frames**: 679 manually annotated frames
- **RoboFlow Link**: https://universe.roboflow.com/kilos-workspace/powerlifting-clean-and-jerk-yz8mx
- **RoboFlow ID**: powerlifting-clean-and-jerk-yz8mx

## Notes & What could have gone better

- Increase dataset size for improved generalization
- Enhance pose localization accuracy
- Reduce confusion between adjacent transition phases

## Authors

- **Jabin K. Wade II** - Department of Computer Science, Prairie View A&M University
- **Jason B. Martin** - Department of Computer Science, Prairie View A&M University
