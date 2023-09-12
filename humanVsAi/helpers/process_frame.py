import EnvironmentConfigurations as EnvConfig 
import cv2
import numpy as np

def process_frame(observation):
    if observation.shape != EnvConfig.EXPECTED_IMAGE_SHAPE:
        raise ValueError(f"Unexpected observation shape. Expected {EnvConfig.EXPECTED_IMAGE_SHAPE}, but got {observation.shape}.")
    
    resized = cv2.resize(observation[40:, 4:-4], None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    return resized