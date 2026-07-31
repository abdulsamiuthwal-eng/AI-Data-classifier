"""
DecodeLabs — Project 2: Data Classification Using AI
Flask Web Application
"""

import json
import os
from flask import Flask, render_template, request, jsonify
from model.train import run_training_pipeline
from model.predict import predict_species

app = Flask(__name__)

# ─────────────────────────────────────────────
# Helper: Load results from disk
# ─────────────────────────────────────────────
def load_results():
    path = 'model/results.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    """Main dashboard."""
    results = load_results()
    trained = results is not None
    return render_template('index.html', results=results, trained=trained)


@app.route('/landing')
def landing():
    """Get Started landing page."""
    return render_template('landing.html')


@app.route('/train', methods=['GET', 'POST'])
def train():
    """Train the model."""
    message = None
    results = None

    if request.method == 'POST':
        try:
            # Change to app directory so relative paths work
            original_dir = os.getcwd()
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_dir)

            results_data = run_training_pipeline()
            results = load_results()
            message = {'type': 'success', 'text': 'Model trained successfully!'}

            os.chdir(original_dir)
        except Exception as e:
            message = {'type': 'error', 'text': f'Training failed: {str(e)}'}

    return render_template('train.html', message=message, results=results)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page."""
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            sepal_length = float(request.form['sepal_length'])
            sepal_width  = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width  = float(request.form['petal_width'])

            original_dir = os.getcwd()
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_dir)

            prediction = predict_species(sepal_length, sepal_width, petal_length, petal_width)
            prediction['inputs'] = {
                'sepal_length': sepal_length,
                'sepal_width':  sepal_width,
                'petal_length': petal_length,
                'petal_width':  petal_width
            }

            os.chdir(original_dir)
        except FileNotFoundError:
            error = 'Model not trained yet. Please train the model first!'
        except ValueError:
            error = 'Invalid input. Please enter valid numbers.'
        except Exception as e:
            error = f'Prediction error: {str(e)}'

    results = load_results()
    return render_template('predict.html', prediction=prediction, error=error, results=results)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for predictions."""
    try:
        data = request.get_json()
        result = predict_species(
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        )
        return jsonify({'status': 'success', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/results')
def api_results():
    """Return training results as JSON."""
    results = load_results()
    if results:
        return jsonify({'status': 'success', 'data': results})
    return jsonify({'status': 'error', 'message': 'Model not trained yet'}), 404


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # Change to script directory so paths work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("DecodeLabs AI Classification App running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
