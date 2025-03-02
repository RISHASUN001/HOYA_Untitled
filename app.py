from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_person():
    data = request.json
    name = data.get('name', 'Unknown')
    
    response = {
        "name": name,
        "description": f"{name} is a good person."
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
