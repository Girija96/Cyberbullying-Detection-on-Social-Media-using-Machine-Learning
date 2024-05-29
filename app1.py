import pickle
import numpy as np
from flask import Flask, request, render_template

# Create Flask app
app = Flask(__name__)
# Load the pickle files
model_T = pickle.load(open("model_T.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
# Define the mapping of prediction values to labels
prediction_labels = {
    0: 'age',
    1: 'ethnicity',
    2: 'gender',
    3: 'not_cyberbullying',
    4: 'other_cyberbullying',
    5: 'religion'
}
# Define suggestions or summaries for each label
suggestions = {
    'age': 'Be mindful of age-related comments. Promote respect for all age groups.',
    'ethnicity': 'Avoid making comments that could be seen as discriminatory or harmful to any ethnicity.',
    'gender': 'Ensure that your language is inclusive and respectful of all genders.',
    'not_cyberbullying': 'This tweet does not appear to contain harmful content. Keep promoting positive and respectful communication.',
    'other_cyberbullying': 'This tweet may contain other forms of cyberbullying. Consider revising your language to be more respectful and kind.',
    'religion': 'Respect all religions and avoid comments that could be offensive to any religious group.',
}


@app.route("/")
def Home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    tweet = request.form['text_value']
    transformed_tweet = vectorizer.transform([tweet])
    prediction = model_T.predict(transformed_tweet)

    # Get the label for the prediction
    prediction_label = prediction_labels.get(prediction[0], "Unknown")

    # Get the suggestion or summary for the prediction label
    suggestion = suggestions.get(prediction_label, "No suggestion available.")

    prediction_text = "The tweet provided is based on {}.".format(prediction_label)

    return render_template("index.html", output_sentiment=prediction_text, suggestion=suggestion, text_value=tweet)


if __name__ == "__main__":
    app.run(debug=True)
