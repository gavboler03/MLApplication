from fakeTrueDatasetModel import *


def main():
    userInputArray = []
    st.title("Fake News Detector")
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
