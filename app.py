from flask import Flask, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pickled objects
with open('grouped.pkl', 'rb') as f:
    grouped = pickle.load(f)

with open('similarity_matrix.pkl', 'rb') as f:
    similarity_matrix = pickle.load(f)

# Define the recommend_nurses function
def recommend_nurses(Zip_Code, top_n=5):
    index = grouped[grouped["Zip_Code"] == Zip_Code].index.values[0]
    similarity_scores = similarity_matrix[index]
    similar_nurses_index = similarity_scores.argsort()[::-1][1:top_n+1]
    similar_nurses = grouped.loc[similar_nurses_index, ["Last_Name", "Overall_Rating","Phone_Number","Gender"]]
    similar_nurses_sorted = similar_nurses.sort_values(by="Overall_Rating", ascending=False)
    return similar_nurses_sorted.to_dict(orient="records")

# Define the Flask route
@app.route('/recommend_nurses/<string:Zip_Code>/<int:top_n>', methods=['GET'])
def get_recommend_nurses(Zip_Code, top_n):
    # Call the recommend_nurses function
    similar_nurses = recommend_nurses(Zip_Code, top_n)
    return jsonify(similar_nurses)

if __name__ == '__main__':
    app.run(debug=True)
