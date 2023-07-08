import streamlit as st
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.stem import WordNetLemmatizer
import re
import string
import pickle


#import models
cv = pickle.load(open("cv_vectorizer.pkl", "rb"))
lda_model = pickle.load(open("lda_model.pkl", "rb"))
stop_words = pickle.load(open("stop_words.pkl","rb"))
tf = pickle.load(open("tf_vectorizer.pkl","rb"))
ml = pickle.load(open("GuassianNB.pkl","rb"))
pun_word = string.punctuation
lamma = WordNetLemmatizer()


# lemmatization
def preprocessing(txt):
  x = txt.lower()
  x = re.sub("\d+[/?]\w+[/?]\w+:|\d+[|]\w+[|]\w+:|\d+[/]\w+[/]\w+[(]\w+[)]:?", "", x)
  x = re.sub("int[a-z]+d$", "interested", x)
  x = re.sub("[\d+-?,'.]", "", x)
  x = [i for i in nltk.word_tokenize(x) if i not in stop_words and len(i)>1 and i not in pun_word]
  x = [lamma.lemmatize(i) for i in x]
  return " ".join(x)

# topic modeling system
def topic_model(txt):
  user_mgs = txt
  x = preprocessing(user_mgs)
  x = cv.transform([x])
  lda_x = lda_model.transform(x)
  tpic = []
  tpc = lambda x : "not interested" if x == 0 else "interested"
  for i,topic in enumerate(lda_x[0]):
    # print("Topic ",i,": ",topic*100,"%")
    tpc_name = tpc(i)
    prc = np.round(topic*100,2)
    tpic.append([tpc_name,prc])
  return tpic


# classification system 
def Status(user):
  x = user
  x = preprocessing(x)
  x = tf.transform([x])
  x = ml.predict(x.toarray())
  if x == 1:
    return "Not Converted"
  else:
    return "Converted"

def main():
    st.title("Task-1 [Topic Modeling]")
    st.write("It provides details about whether the customer showing interest or not in percentage")
    user = st.text_area("Paste your chats/mgs here for topic modeling")
    if st.button("Model the Topic"):
        topics = topic_model(user)
        for i in topics:
            st.write(i)

    st.title("Task-2 [Classification System]")
    st.write("It tells whether the person is converted or not for the above mentioned text")
    # user2 = st.text_area("Paste your chats/conversation here")
    if st.button("Predict"):
      result = Status(user)
      if result == "Converted":
        st.success(result)
      else:
        st.error(result)





    st.write("## Thank you for Visiting \nProject by Madhukumar G")
    st.markdown("<h1 style='text-align: right; color: #d7e3fc; font-size: small;'><a href='https://github.com/madhumarvel123/CRM_NLP_PROJECT'>Source Code link?</a></h1>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
