from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome! Backend is working."})

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Backend is connected Yay - Trial _jan successfully!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
