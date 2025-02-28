from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Backend is working!"})

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"message": "Connected to backend successfully!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
