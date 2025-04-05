from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import subprocess
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
RESULTS_FOLDER = './tests/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(audio_file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(filepath)

    # Run automation.py with the uploaded audio
    try:
        result = subprocess.run(['python3', '/Users/aryangupta/Desktop/UIUC/CURRENT/CS-444/PitchPal-CS444/src/automation.py', filepath], check=True)
    except subprocess.CalledProcessError:
        return jsonify({"error": "Failed to process audio"}), 500

    # Return list of result JSONs
    output_files = [f for f in os.listdir(RESULTS_FOLDER) if f.endswith('.json')]
    return jsonify({"message": "Analysis complete", "results": output_files})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
