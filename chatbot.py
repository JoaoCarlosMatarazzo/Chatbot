import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random
from flask import Flask, request, jsonify

# Baixar pacotes necessários do nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Dados
data = {
    "qual o seu nome?": ["Eu sou Chatbot."],
    "como voce esta?": ["Estou bem, obrigado por perguntar!"],
    "o que voce pode fazer?": ["Eu posso responder perguntas simples e interagir com voce."],
}

# Função para treinar o modelo
def train_model(data):
    questions = list(data.keys())
    answers = list(data.values())
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(questions, [i for i in range(len(answers))])
    return model, answers
model, answers = train_model(data)

def get_response(model, answers, user_input):
    # Prever a resposta
    pred = model.predict([user_input])[0]
    return random.choice(answers[pred])

# Configurar a aplicação Flask
app = Flask(__name__)
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = get_response(model, answers, user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
