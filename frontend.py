from fakeTrueDatasetModel import *


def main():
    userInputArray = []
    st.title("Fake News Article Detector")
    st.header("This is a news article detector and not fact checker")
    st.subheader("For single line entries or entries such as 'George Washington was the first president of the USA.', it will return false.")
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
