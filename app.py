from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    message = ''
    if request.method == "POST":
        if 'clear' in request.form:
            message = ''
            result = None
        elif 'submit' in request.form:
            message = request.form["message"]
            data = vectorizer.transform([message])
            prediction = model.predict(data)[0]
            result = "✅ Safe message" if prediction == 0 else "⚠️ SPAM message!"
    return render_template("index.html", result=result, message=message)

if __name__ == "__main__":
    app.run(debug=True)
