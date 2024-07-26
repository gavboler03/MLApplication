import streamlit as st
from fakeTrueDatasetModel import pac, tfidf_vectorizer
import time

# # Simulate model training
# def train_model():
#     time.sleep(10)  
#     return True

# # Placeholder to check if the model is loaded
# if 'model_loaded' not in st.session_state:
#     st.session_state.model_loaded = False

# # Load the model if not already loaded
# if not st.session_state.model_loaded:
#     st.title("Fake News Article Detector") 
#     with st.spinner('Model is loading and training, please wait...'):
#         if train_model():
#             st.session_state.model_loaded = True
#             st.rerun()  # Rerun the app once the model is loaded
# else:
def main():
        userInputArray = []
        st.title("Fake News Article Detector")
        st.caption("This is a fake news article detector and not fact checker.")
        st.caption('Entries like "Washington was the first president" will return false.')
        with st.spinner("Model is loading. Please wait..."):
            time.sleep(10)
        userInput = st.text_input("Paste a news article here.")
        answer = ""

        if st.button("Submit"):
            userInputArray.append(userInput)
            userInput = tfidf_vectorizer.transform(userInputArray)
            answer = pac.predict(userInput)

            if answer == 0:
                answer = "Fake News"
            else:
                answer = "Real News"

        st.success(answer)

main()
