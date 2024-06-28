from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('sentiment_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/', methods=['GET'])
def get():
    return jsonify({
        'statusCode': 200,
        'message': 'Server running successfully'
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data['input']
    sequence = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequence, maxlen=10, padding='post')
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = 'positive' if prediction > 0.5 else 'negative'
    return jsonify({'prediction': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
