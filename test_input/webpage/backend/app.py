from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import tempfile
from PIL import Image
from rembg import remove
import numpy as np

app = Flask(__name__)
CORS(app)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return send_from_directory('../frontend/public', 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/api/remove-background', methods=['POST'])
def remove_background():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        # Create temporary files for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_input, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_output:
            
            # Save uploaded file
            file.save(temp_input.name)
            
            # Process image
            input_image = Image.open(temp_input.name)
            output_image = remove(input_image)
            
            # Get background color from request
            bg_color = request.form.get('background_color', '#FFFFFF')
            if bg_color.startswith('#'):
                bg_color = bg_color[1:]
            r, g, b = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))
            
            # Create new image with background color
            background = Image.new('RGB', output_image.size, (r, g, b))
            background.paste(output_image, mask=output_image.split()[3])
            
            # Save result
            background.save(temp_output.name, 'PNG')
            
            # Return processed image
            return send_file(temp_output.name, mimetype='image/png')
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Cleanup temporary files
        try:
            os.unlink(temp_input.name)
            os.unlink(temp_output.name)
        except:
            pass


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)