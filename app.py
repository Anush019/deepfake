import os
import cv2
from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16 MB

# Load the deepfake detection model
model_path = "./models"  # Replace with your model directory
pipe = pipeline("image-classification", model=model_path)

def detect_deepfake_from_video(video_path, frame_skip=30, score_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    deepfake_count = 0
    real_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            temp_frame_path = "temp_frame.jpg"
            cv2.imwrite(temp_frame_path, rgb_frame)

            results = pipe(temp_frame_path)
            max_score = 0
            label = "Unknown"
            
            for result in results:
                if result['score'] > max_score:
                    max_score = result['score']
                    label = result['label']

            if label == 'Fake' and max_score > score_threshold:
                deepfake_count += 1
            elif label == 'Real' and max_score > score_threshold:
                real_count += 1

            os.remove(temp_frame_path)

        frame_count += 1

    cap.release()
    return frame_count, deepfake_count, real_count
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        if file:
            filename = file.filename  # Get the filename directly from the file variable
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                frame_count, deepfake_count, real_count = detect_deepfake_from_video(file_path)

                # Determine if the video is fake or real
                classification_result = "Real" if deepfake_count == 0 else "Fake"
                
                # Pass all the necessary data to the result template
                return render_template('result.html', 
                                       frame_count=frame_count, 
                                       deepfake_count=deepfake_count, 
                                       real_count=real_count,
                                       filename=filename,
                                       classification_result=classification_result)  # New variable for classification

            else:
                return "Unsupported file type. Please upload a video file."

    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create uploads directory if it doesn't exist
    app.run(debug=True)
