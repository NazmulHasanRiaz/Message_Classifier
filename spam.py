import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# We have to perform four steps to execute ...
# 1.Preprocess.
# 2.Vectorize.
# 3.predict.
# 4.Display.

###############################
# 1.Preprocess.
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)  # word_tokenize makes split the text into word

    # Removing special cherecter ...

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # Removig stop words and puntuation ...

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # stemming

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorize.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('SMS Spam Classifier')
input_sms = st.text_area('Enter message')
if st.button('Predict'):

    transform_sms = transform_text(input_sms)

    # 2.Vectorize.
    vectorize_input = tfidf.transform([transform_sms])

    # 3.Predict
    result = model.predict(vectorize_input)[0]

    #4. Display
    if result == 1:
        st.header("OPP's ! it's a Spam message ...")
    else:
        st.header("It's not Spam message ...")