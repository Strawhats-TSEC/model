# app.py

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the pre-trained BERT model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Create a dataset of strings
dataset = ["This is a string", "This is another string", "This is a similar string"]

@app.route('/find_most_similar_strings', methods=['POST'])
def find_most_similar():
    # Get the input string from the request
    input_string = request.json['input_string']

    # Generate embeddings for the input string and the dataset strings
    input_embedding = model.encode(input_string)
    dataset_embeddings = model.encode(dataset)

    # Calculate the cosine similarity between the input string and the dataset strings
    similarities = cosine_similarity([input_embedding], dataset_embeddings)[0]

    # Find the indices of the most similar strings
    most_similar_indices = similarities.argsort()[-3:][::-1]

    # Return the most similar strings
    return jsonify([dataset[i] for i in most_similar_indices])

if __name__ == '__main__':
    app.run()