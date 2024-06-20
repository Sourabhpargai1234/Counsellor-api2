# Import necessary modules
import sys
import json
from flask import Flask, request, jsonify
#from transformers import pipeline
from gradientai import Gradient
from gradientai.openapi.client.exceptions import UnauthorizedException
from flask_cors import CORS
from dotenv import load_dotenv
import os


# Load the model once at the start
#qa = pipeline('question-answering')

# Initialize Flask application
app = Flask(__name__)
CORS(app)

load_dotenv()

GRADIENT_ACCESS_TOKEN = os.getenv('GRADIENT_ACCESS_TOKEN')
GRADIENT_WORKSPACE_ID = os.getenv('GRADIENT_WORKSPACE_ID')

# Define endpoint for question answering
'''@app.route('/qa', methods=['POST'])
def qa_endpoint():
    # Get JSON data from the request body
    data = request.get_json()

    # Extract 'context' and 'question' from JSON data
    context = data.get('context')
    question = data.get('question')

    if not context or not question:
        return jsonify({'error': 'Both context and question parameters are required.'}), 400

    # Perform question answering
    result = qa(context=context, question=question)

    # Return the result as JSON response
    return jsonify(result)'''

@app.route('/', methods=['GET'])
def success():
    data = {'message': 'Server set-up successful', 'status': 'OK'}
    return jsonify(data)

@app.route('/llm', methods=['POST'])
def llm_endpoint():
    try:
        # Get JSON data from the POST request
        data = request.json
        if 'question' not in data:
            return jsonify({'error': 'Missing question parameter'}), 400

        question = data['question']

        with Gradient(access_token=GRADIENT_ACCESS_TOKEN) as gradient:
            base_model = gradient.get_base_model(base_model_slug="nous-hermes2")

            new_model_adapter = base_model.create_model_adapter(
                name="test model 3"
            )
            sample_query = f"### Instruction: {question} \n\n### Response:"
            print(f"Asking: {sample_query}")

            samples = [
                { "inputs": "### Instruction: What is your branch \n\n### Response: Btech" },
                { "inputs": "### Instruction: Skills ? \n\n### Response: Coding, Development" },
                { "inputs": "### Instruction: Job? \n\n### Response: SDE at google" },
                { "inputs": "### Instruction: What is your branch \n\n### Response: Mechanical Engineering" },
            ]

            new_model_adapter.fine_tune(samples=samples)

            completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
            print(f"Generated (after fine-tune): {completion}")

            new_model_adapter.delete()

        return jsonify({'generated_output': completion})

    except UnauthorizedException as e:
        return jsonify({'error': 'Unauthorized: Check your API key and permissions.'}), 401
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
 

if __name__ == '__main__':
    # Run Flask web server
    app.run(debug=True)
