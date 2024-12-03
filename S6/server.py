from flask import Flask, render_template
import json
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    with open('loss_log.json', 'r') as f:
        loss_data = json.load(f)
    return render_template('index.html', loss_data=loss_data)

if __name__ == "__main__":
    app.run(debug=True) 