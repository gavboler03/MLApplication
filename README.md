# Machine Learning Application: Fake News Detector

## Project Summary

Fake news articles circulating are one of the main issues with media in the 21st century. Whether by malicious intent or ignorance, there is a fair bit of fake news articles that makes rounds through social media like Facebook and Twitter. We will tackle this problem by creating a model that can distinguish fake news articles from real news headlines. Sometimes all it may take is a headline to cause mass anger. In a way to try and curve this, Samuel Flowers and Gavin Boler for a machine learning class project decided to create a machine learning model that can help distinguish between the two. As a reminder, this is not a fact checker and is not a search engine. This is just a model that determines what may be a real news article, such as from the BBC, and what is a fake article, such as from the Onion. 

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
5. Go to web browser and navigate to the address given in the command terminal.
6. Input the contents of a news article into the prompt.
7. Click the submit button. The output will either be `This news is real` or `This news is fake.`
8. Refresh the page if you want to input another news article
9. When finished, enter `Ctrl + C` to stop the docker container and `docker compose down` to bring down the docker container.

## Results

Initial model test had a 99% accuracy on the training data.
