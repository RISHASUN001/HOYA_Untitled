from flask import Flask, request, jsonify

app = Flask(__name__)

# Route for checking if the API is running
@app.route('/', methods=['GET'])
def home():
    return "Flask API is running!", 200

# POST route for processing input
@app.route('/', methods=['POST'])
def analyze_person():
    data = request.json
    name = data.get('name', 'Unknown')
    
    response = {
        "name": name,
        "description": f"{name} is a good person."
    }
    
    print(f"Received request for: {name}")  # Print statement for logging
    
    return jsonify(response)

if __name__ == '__main__':
    print("Flask app is starting...")  # Health check message
    app.run(host='0.0.0.0', port=3000)
