import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import pickle
from data.load_datasets import get_diamonds, get_wines
from flask import Flask, request, jsonify


app = Flask(__name__)
@app.route("/")
def index():
    paths = [str(rule) for rule in app.url_map.iter_rules()]
    paths.remove('/static/<path:filename>')
    return "<br>".join(paths)

@app.route('/predict', methods=['POST'])
def predict():
    pass

@app.route('/train', methods=['POST'])
def train():
    pass

@app.route('/load', methods=['POST'])
def load():
    pass

def main():
    app.run(host='0.0.0.0', port=5000, debug=True)
    
    # diamonds_df = get_diamonds()
    # wines_df = get_wines()

    # diamonds_df.info()
    # wines_df.info()


if __name__ == "__main__":
    main()