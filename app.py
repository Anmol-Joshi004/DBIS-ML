from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open('_model.pkl', 'rb') as model_file:
    Naive_Bayes_clf = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        email_content = request.form['email_content']
        email_vectorized = vectorizer.transform([email_content])
        prediction = Naive_Bayes_clf.predict(email_vectorized)[0]
        prediction = 'Spam' if prediction == 1 else 'Not Spam'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
