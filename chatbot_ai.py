from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

questions = [
"hi",
"hello",
"what courses are available",
"how can I take admission",
"do you provide placement",
"bye"
]

answers = [
"Hello! I am your AI Student Assistant.",
"Hello! How can I help you?",
"We offer AI, Data Science and Web Development courses.",
"You can apply online through the college website.",
"Yes, we provide placement assistance.",
"Goodbye! Have a great day."
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def get_response(user_input):

    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()

    return answers[index]