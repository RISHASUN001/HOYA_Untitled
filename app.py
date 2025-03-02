from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Route for checking if the API is running
@app.route('/', methods=['GET'])
def home():
    return "Flask API is running!", 200

# Handle OPTIONS requests for CORS preflight
@app.route('/', methods=['OPTIONS'])
def options():
    response = jsonify({})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    return response, 200

# POST route for processing input
@app.route('/', methods=['POST'])
def analyze_person():
    data = request.json
    name = data.get('name', 'Unknown')
    
    response = {
        "name": name,
        "description": f"{name} is a good person. yayyyy!!!"
    }
    
    print(f"Received request for: {name}")  # Print statement for logging
    
    return jsonify(response)

if __name__ == '__main__':
    print("Flask app is starting...")  # Health check message
    app.run(host='0.0.0.0', port=3000)
