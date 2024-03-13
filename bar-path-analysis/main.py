import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import cv2

img = cv2.imread(r"C:\Users\COMPUBROTHER\Documents\DATA_ANALYTICS\Projects\sports-performance\pose-estimation-analysis\Capture.PNG")
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

model_path = r"C:\Users\COMPUBROTHER\Documents\DATA_ANALYTICS\Projects\sports-performance\pose-estimation-analysis\pose_landmarker_heavy.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

detector = vision.PoseLandmarker.create_from_options(options)

img = mp.Image.create_from_file("Capture.PNG")

detection_result = detector.detect(img)
annotated_image = draw_landmarks_on_image(img.numpy_view(), detection_result)
cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
cv2.imshow(visualized_mask)