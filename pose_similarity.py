import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple
import time
import tkinter as tk
from tkinter import filedialog

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

def select_video_file():
    """Open a file dialog to select a video file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select Reference Video",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )
    return file_path

def main():
    # Initialize
    try:
        pose_detector = PoseSimilarity()
        print("Initializing webcam...")
        cap = cv2.VideoCapture(0)  # Live video from webcam
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Open file dialog to select reference video
        print("Please select a reference video file...")
        reference_video_path = select_video_file()
        
        if not reference_video_path:  # If user cancels selection
            print("No video file selected. Exiting...")
            return
            
        print(f"Loading reference video: {reference_video_path}")
        ref_cap = cv2.VideoCapture(reference_video_path)
        
        if not ref_cap.isOpened():
            print(f"Error: Could not open reference video at {reference_video_path}")
            return
            
        # Get total frames in reference video
        total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process and store all reference video frames
        print("Processing reference video...")
        reference_sequence = []
        while True:
            ret, ref_frame = ref_cap.read()
            if not ret:
                break
                
            landmarks = pose_detector.extract_landmarks(ref_frame)
            reference_sequence.append(landmarks)
            
        if not reference_sequence:
            print("Error: Could not process reference video")
            return
            
        print(f"Reference video processed: {len(reference_sequence)} frames")
        print("Setup complete! Press 'q' to quit, 'r' to restart sequence matching")
    except Exception as e:
        print(f"An error occurred during setup: {str(e)}")
        return
    
    try:
        while True:
            # Read live frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract landmarks from live frame
            live_landmarks = pose_detector.extract_landmarks(frame)
            
            # Calculate and display similarity
            if live_landmarks is not None:
                # Find best matching pose in the reference sequence
                similarity, match_idx = pose_detector.find_best_match(live_landmarks, reference_sequence)
                
                # Get the corresponding reference frame for display
                if match_idx >= 0:
                    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, match_idx)
                    ret, current_ref_frame = ref_cap.read()
                    if not ret:
                        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, current_ref_frame = ref_cap.read()
                    
                    if ret:
                        # Draw pose landmarks
                        frame = pose_detector.draw_pose(frame)
                        ref_frame_with_pose = pose_detector.draw_pose(current_ref_frame)
                        
                        # Show reference frame and live frame side by side
                        ref_frame_resized = cv2.resize(ref_frame_with_pose, (frame.shape[1]//2, frame.shape[0]//2))
                        frame_resized = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
                        
                        # Create a horizontal stack of the two frames
                        combined_frame = np.hstack((ref_frame_resized, frame_resized))
                    else:
                        combined_frame = frame
                        
                    pose_detector.draw_similarity_text(combined_frame, similarity, match_idx, total_frames)
                else:
                    # If no match found, just show the live feed
                    frame = pose_detector.draw_pose(frame)
                    combined_frame = frame
                    pose_detector.draw_similarity_text(combined_frame, similarity)
            else:
                # If no pose detected, just show the live feed
                frame = pose_detector.draw_pose(frame)
                combined_frame = frame
            
            # Display the result
            cv2.imshow('Pose Similarity Comparison', combined_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Clean up
        cap.release()
        ref_cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
