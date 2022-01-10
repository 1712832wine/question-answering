import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_data(filename):
    with open(filename, encoding="utf8") as f:
        data = f.read()
    return data


def normalise_text(text):
    text = re.sub("&\w+;", "", text)  # remove punctuation encoding
    text = re.sub("&#\d+;", "", text)

    text = re.sub("gt", "", text)  # remove punctuation encoding
    text = re.sub("em", "", text)
    text = re.sub("lt", "", text)

    text = text.lower()  # lowercase
    text = text.replace(r"http\S+", "URL")  # remove URL addresses
    text = text.translate(str.maketrans(
        '', '', string.punctuation))  # remove punctuations
    text = re.sub(r"\s{2,}", " ", text)  # replace >= 2 spaces
    text = re.sub(r"^\s", "", text)  # remove starting spaces
    text = re.sub(r"\s$", "", text)  # remove ending spaces

    return text


def get_questions(text):
    questions = []
    pattern = re.compile(r'- -.*\?')
    matches = pattern.finditer(text)
    for match in matches:
        s = match.group(0)
        s = normalise_text(s)
        questions.append(s)
    return questions


def get_answers(text):
    answers = []
    pattern = re.compile(r'  - .*\n')
    matches = pattern.finditer(text)
    for match in matches:
        s = match.group(0)
        s = normalise_text(s)
        answers.append(s)
    return answers


def tf_idf(data_src, data_dest, question):
    d = {'Question': data_src, 'Answer': data_dest}
    df = pd.DataFrame(data=d)

    vectorizer = TfidfVectorizer()
    # fit questions and answers to the vectorizer
    vectorizer.fit(np.concatenate((df.Question, df.Answer)))
    # transform questions into vectorizers
    Question_vectors = vectorizer.transform(df.Question)
    # transform question into vectorizer
    input_question_vector = vectorizer.transform([question])
    # compute similarity
    similarities = cosine_similarity(input_question_vector, Question_vectors)
    # get similarities max
    closest = np.argmax(similarities, axis=1)
    document = df.Answer.iloc[closest].values[0]

    # chú ý trả về 500 kí tự đầu
    return document[:500]


def find_docs(question):
    text = read_data('vaccine.txt')
    data_src = get_questions(text)
    data_dest = get_answers(text)
    document = tf_idf(data_src, data_dest, question)
    print("document: ", document)
    return document
