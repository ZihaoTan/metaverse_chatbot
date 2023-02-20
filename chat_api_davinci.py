import os
import openai
from flask import Flask, request, jsonify, make_response
import numpy as np
import pickle
import json
import pandas as pd
import logger

app = Flask(__name__)

def change_object_to_cat(df, cat_feature):
    for c in df[cat_feature]:
        col_type = df[c].dtype
        if col_type == 'object' or col_type.name == 'category':
            df[c] = df[c].astype('category')

    return df

with open('cost_prediction_model_with_metverse_usecase.pkl', 'rb') as f:
    model = pickle.load(f)
feature = ['feat_debit_location', 'feat_bene_location', 'feat_currency']
print('loaded model and feature')

@app.route("/message", methods=['POST'])
def get_answer():
    data = request.get_json(force=True)
    question = data['question']
    if 'ost prediction' in question:
        l = question.split(' ')
        from_location = l[l.index('from') + 1]
        to_location = l[l.index('to') + 1]
        currency = l[l.index('in') + 1]
        d = {'feat_debit_location': from_location, 'feat_bene_location': to_location, 'feat_currency': currency}
        df = pd.DataFrame([d])
        df = change_object_to_cat(df, feature)
        y_pred = model.predict(df, num_iteration=model.best_iteration)
        text = f'Cost from {from_location} to {to_location} in {currency} is {y_pred[0]}'

    else:
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=question,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
        )
        app.logger.debug(f'User input: {question}')
        text = response["choices"][0]["text"]
        app.logger.debug(f'Chatbot response: {text}')

    return json.dumps({'text': text})

@app.route("/home", methods=["GET"])
def hello():
    return "welcome"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3001, debug=True)
