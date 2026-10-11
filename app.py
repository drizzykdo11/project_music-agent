from flask import Flask, render_template, request
from agent import answer_question

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    answer = answer_question(question)
    return render_template('index1.html', answer=answer, question=question)

if __name__ == '__main__':
    app.run(debug=True)