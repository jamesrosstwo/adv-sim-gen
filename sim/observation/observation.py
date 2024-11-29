import cv2
import numpy as np

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    normalized = resized / 255.0
    return normalized

stack_size = 4
stacked_frames = deque([np.zeros((84, 84), dtype=np.float32) for _ in range(stack_size)], maxlen=stack_size)

def stack_frames(stacked_frames, new_frame):
    processed_frame = preprocess_frame(new_frame)
    stacked_frames.append(processed_frame)
    return np.stack(stacked_frames, axis=2)
