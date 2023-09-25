import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

df = pd.read_csv('Provider Rating Data.csv')

df.head()

df.info()

df = df.rename(columns={'Zip Code': 'Zip_Code', 'Overall Rating': 'Overall_Rating','Last Name':'Last_Name','Phone Number':'Phone_Number'})

df.head()

df["Zip_Code"] = df["Zip_Code"].astype(str)

# Step 2: Grouping the data by zip code and rating
grouped = df.groupby(["Zip_Code", "Overall_Rating"])["Last_Name", "Phone_Number","Gender"] \
          .agg({"Last_Name": list, "Phone_Number": list,'Gender':list}).reset_index()

grouped["avg_rating"] = grouped.groupby("Zip_Code")["Overall_Rating"].transform("mean")

# Step 3: Calculating similarity between nurses
features = grouped[["Zip_Code", "Overall_Rating", "avg_rating"]]
X = pd.get_dummies(features)
similarity_matrix = cosine_similarity(X)

def recommend_nurses(Zip_Code, top_n=5):
    index = grouped[grouped["Zip_Code"] == Zip_Code].index.values[0]
    similarity_scores = similarity_matrix[index]
    similar_nurses_index = similarity_scores.argsort()[::-1][1:top_n+1]
    similar_nurses = grouped.loc[similar_nurses_index, ["Last_Name", "Overall_Rating","Phone_Number","Gender"]]
    similar_nurses_sorted = similar_nurses.sort_values(by="Overall_Rating", ascending=False)
    return similar_nurses_sorted.to_dict(orient="records")


# Call the recommend_doctors function
#similar_nurses = recommend_nurses("305183480",top_n=5)
#print(similar_nurses)

#Pickle Dump

# Save the model
#with open('model.pkl', 'wb') as f:
 #  pickle.dump(recommend_nurses, f)

# Save the objects
with open('grouped.pkl', 'wb') as f:
    pickle.dump(grouped, f)

with open('similarity_matrix.pkl', 'wb') as f:
    pickle.dump(similarity_matrix, f)

with open('recommend_nurses.pkl', 'wb') as f:
    pickle.dump(recommend_nurses, f)



