# Machine Learning Application: Fake News Detector

## Project Summary

Fake news circulating is one of the main issues with media in the 21st century. Whether by malicious intent or ignorance, there is a fair bit of fake news that makes rounds through social media like Facebook and Twitter. We will tackle this problem by creating a model that can distinguish fake news from real news headlines. Sometimes all it may take is a headline to cause mass anger.

## Approach

A PassiveAgressive Classifier algorithm could be used to train the model. In this algorithm, if the model makes a correct prediction, nothing is updated to the model. However, if there is an incorrect prediction, the model is changed to make it correct. Additionally, a TfidfVectorizer can be used to calculate the frequency of certain words/names to train the model. The TfidfVectorizer will check the frequency of words in the dataset. An alternative or addition to the TfidfVectorizer would be the Decision Tree. There may be some trial and error periods testing different models and tracking performance based on each model to make the most efficient decision. Lastly, a docker container will be used at the end of the project.

## Resources

Many sources discussing this use some form of tree such as a decision tree or random forest. We learned about Passive-Aggresive algorithms from GeeksforGeeks. This will allow us to train our model which is processing over 6,000 news headlines. There is a possibility that there will be a combined dataset containing over 20,000 unique headlines.

## GitHub Link

https://github.com/gavboler03/MLApplication

## How to Use This Project

1. Install Docker Desktop.
2. Open the Terminal.
3. Navigate to your working directory with `cd path/to/working/directory`.
4. Run the command `docker compose up --build`.
5. Go to web browser and navigate to `http://localhost:8501`.
