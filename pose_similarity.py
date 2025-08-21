import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple
import time

class PoseSimilarity:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.landmark_names = [
            name for name in self.mp_pose.PoseLandmark.__dict__.keys() 
            if not name.startswith('_')
        ]

    def extract_landmarks(self, frame) -> List[float]:
        """Extract pose landmarks from a frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmark coordinates and flatten them
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return landmarks

    def calculate_similarity(self, landmarks1: List[float], landmarks2: List[float]) -> float:
        """Calculate cosine similarity between two sets of landmarks."""
        if landmarks1 is None or landmarks2 is None:
            return 0.0
            
        # Convert to numpy arrays
        vec1 = np.array(landmarks1)
        vec2 = np.array(landmarks2)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        # Convert to percentage
        return max(0, min(100, similarity * 100))
    
    def find_best_match(self, current_landmarks: List[float], reference_sequence: List[List[float]], window_size: int = 5) -> Tuple[float, int]:
        """Find the best matching pose in the reference sequence."""
        if not reference_sequence or current_landmarks is None:
            return 0.0, -1
        
        max_similarity = 0.0
        best_match_idx = -1
        
        # Look at the recent frames in the reference sequence
        start_idx = max(0, len(reference_sequence) - window_size)
        for idx in range(start_idx, len(reference_sequence)):
            ref_landmarks = reference_sequence[idx]
            if ref_landmarks is not None:
                similarity = self.calculate_similarity(current_landmarks, ref_landmarks)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_idx = idx
                    
        return max_similarity, best_match_idx

    def draw_similarity_text(self, frame, similarity: float, frame_match: int = -1, total_frames: int = 0):
        """Draw similarity percentage and progress on frame."""
        # Draw similarity percentage
        text = f"Similarity: {similarity:.1f}%"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw progress indicator if we have frame match information
        if frame_match >= 0 and total_frames > 0:
            progress = f"Progress: {frame_match}/{total_frames}"
            cv2.putText(frame, progress, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw progress bar
            bar_width = int(frame.shape[1] * 0.8)
            bar_height = 20
            x = int((frame.shape[1] - bar_width) / 2)
            y = 100
            
            # Background bar (gray)
            cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (128, 128, 128), -1)
            
            # Progress bar (green)
            progress_width = int(bar_width * (frame_match / total_frames))
            cv2.rectangle(frame, (x, y), (x + progress_width, y + bar_height), (0, 255, 0), -1)

    def draw_pose(self, frame):
        """Draw pose landmarks on frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        
        return frame


