from flask import Flask, render_template, Response, request
import cv2
from pose_similarity import PoseSimilarity
import os

app = Flask(__name__)
pose_detector = None
reference_sequence = None
total_frames = 0

def generate_frames():
    global pose_detector, reference_sequence, total_frames
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame and detect pose
        if pose_detector and reference_sequence:
            live_landmarks = pose_detector.extract_landmarks(frame)
            
            if live_landmarks is not None:
                # Find best matching pose
                similarity, match_idx = pose_detector.find_best_match(live_landmarks, reference_sequence)
                frame = pose_detector.draw_pose(frame)
                pose_detector.draw_similarity_text(frame, similarity, match_idx, total_frames)

        # Convert frame to bytes
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_video():
    global pose_detector, reference_sequence, total_frames
    
    if 'video' not in request.files:
        return 'No video file uploaded', 400
        
    video_file = request.files['video']
    if video_file.filename == '':
        return 'No selected file', 400
        
    # Save the uploaded video temporarily
    temp_path = 'temp_video.mp4'
    video_file.save(temp_path)
    
    # Initialize pose detector if not already done
    if pose_detector is None:
        pose_detector = PoseSimilarity()
    
    # Process the reference video
    ref_cap = cv2.VideoCapture(temp_path)
    total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    reference_sequence = []
    while True:
        ret, ref_frame = ref_cap.read()
        if not ret:
            break
        landmarks = pose_detector.extract_landmarks(ref_frame)
        reference_sequence.append(landmarks)
    
    ref_cap.release()
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return 'Video processed successfully', 200

if __name__ == '__main__':
    app.run(debug=True)
