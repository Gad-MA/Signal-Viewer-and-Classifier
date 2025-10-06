from flask import Flask, request, jsonify, send_file
from doppler.doppler_sound_generator import doppler_effect_wav_generator
from doppler.doppler_detector_and_velocity_and_frequency import analyze_doppler_wav
import os

app = Flask(__name__)

@app.route('/generate-doppler', methods=['POST'])
def generate_doppler():
    try:
        # Extract parameters from the request JSON
        data = request.get_json()
        
        frequency = data.get('frequency')
        velocity = data.get('velocity')
        distance = data.get('normal_distance')
        half_simulation_duration = data.get('half_simulation_duration')

        # Validate parameters
        if not all([frequency, velocity, distance, half_simulation_duration]):
            return jsonify({"error": "Missing required parameters."}), 400

        # Call the doppler_effect_wav_generator function
        doppler_effect_wav_generator(source_velocity=velocity, source_freq=frequency, normal_distance=distance, half_simulation_duration=half_simulation_duration)

        return jsonify({"message": "Doppler effect WAV file generated successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-doppler-sound', methods=['POST'])
def generate_doppler_sound():
    try:
        # Get parameters from JSON request
        data = request.get_json()
        
        # Extract and validate parameters with defaults
        source_velocity = float(data.get('source_velocity', 20))
        source_freq = float(data.get('source_freq', 150))
        normal_distance = float(data.get('normal_distance', 10))
        half_simulation_duration = float(data.get('half_simulation_duration', 3))

        # Generate the WAV file
        doppler_effect_wav_generator(
            source_velocity=source_velocity,
            source_freq=source_freq,
            normal_distance=normal_distance,
            half_simulation_duration=half_simulation_duration
        )

        # Get the path to the generated WAV file
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        wav_file = os.path.join(script_dir, 'doppler', 'doppler_effect.wav')

        # Return the WAV file
        return send_file(
            wav_file,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='doppler_effect.wav'
        )

    except Exception as e:
        return {'error': str(e)}, 400

@app.route('/analyze-doppler', methods=['POST'])
def analyze_doppler():
    try:
        # Get parameters from request
        data = request.get_json()
        distance = float(data.get('distance', 10))  # Default distance is 10m
        
        # Get the path to the WAV file
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        wav_file = os.path.join(script_dir, 'doppler', 'doppler_effect.wav')
        
        # Analyze the WAV file
        results = analyze_doppler_wav(wav_file, d=distance, visualize=False)
        
        # Prepare response
        response = {
            'has_doppler_effect': results['has_doppler_effect'],
            'message': 'Doppler effect detected!' if results['has_doppler_effect'] else 'No Doppler effect detected.'
        }
        
        if results['has_doppler_effect']:
            response.update({
                'estimated_source_frequency': float(results['source_freq']),
                'estimated_source_velocity': float(results['source_velocity'])
            })
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)