# Event or Non-Event

![class_imbalance_graph.jpeg](./images/bigram_positive_wordcloud_graph.png
)

**Authors:** [Michael Wirtz](https://github.com/mwirtz946), [Aidan Coco](https://github.com/acoco10)

## Overview

This project analyzes the needs of The New York Times. The firm is pursuing tweets as a possible avenue for new stories about a certain set of keywords. We defined the positve and negative classes as Event and Non-Event, respectively. The differentiation between the positive and negative class in the dataset was low according to our exploratory data analysis. Our final results improved upon the baseline model by about 8% for the F1 Score. We finished with a model that had an F1 Score of 0.61. We 

## Business Problem

The New York Times is looking to find possible stories on Twitter using a specific set of keywords. They quickly run into the problem of dual-meanings and an overload of query results. They have tasked a team to help them identify when a tweet is an event or not an event. Instead of having staff manually go through all tweets to find events, they want to possess a classifier that can automatically perform this duty.  

## Data

In order to help the New York Times, we utilized a [Kaggle dataset](https://www.kaggle.com/vstepanenko/disaster-tweets) possessing keyword query results and their corresponding labels (event or non-event). The data includes 11,000 tweets, with 219 keywords. Since there were so many keywords, some of the keywords only accounted for a very small number of observations in the dataset.

Here's the most frequently occuring keywords in both the positive and negative classes:

![class_imbalance_graph.jpeg](./images/event_keywords_graph.png
)

![class_imbalance_graph.jpeg](./images/non_event_keywords_graph.png
)

The dataset also includes general location information, of which there were many missing values. Class imbalance within the dataset was higher, skewing heavily to the negative class. This is probably because the text data between the two classes is incredibly similar, so we had to look for small differences in our EDA process. 

![hyperlink_graph.png](./images/hyperlink_graph.png
)

The original classification of the dataset is disaster (positive class) and no disaster (negative class). 
When searching through the dataset's tweets and labels, we found that the data was less specific to disasters and could be more accurately be defined as event vs non-event. Therefore, we used this data to produce a model for the needs of the New York Times. 

Here's how we defined each: 

<table>
<tr>
<th> Target Classes</th>
</tr>
<tr>
<td>

<ul>
<li> <b>Event (1)</b>: tweets which factually describe a real and noteworthy occurance.</li>
<li> <b>Non-Event (0)</b>: tweets which do not factually describe a real and noteworthy occurance.</li>
</ul>

</td>
</tr>
</table>


## Methods

Before we could do anything else, we needed to decide what evaluation metric would be most appropriate for the business problem. We decided that F1 Score would be perfect, as it is a measure of the model's ability to differentiate between the two classes. You can see in the following wordclouds that this task was not simple. These are the most used words in the positive and negative classes:

![hyperlink_graph.png](./images/unigram_positive_wordcloud_graph.png
)

![hyperlink_graph.png](./images/unigram_negative_wordcloud_graph.png
)

Then, prior to modeling, we lemmitized the words in each tweets to increase the comparability of the tweets. From there we used TFIDF Vectorizer to get the relative uniqueness for each word in each tweet. 

We used KNN as a baseline classification model, which produced an F1 Score of 0.5044 and an Accuracy Score of 0.8773. 

From there, we used a Random Forest Classifier and a Naive Bayes Classifier to improve upon our baseline KNN model. 

Finally, since there were so many keywords in the dataset, we used our final model to predict on a dataset that we manually labeled. This dataset contained only the top 10 keywords for the positive class and the top 10 keywords for the negative class. 

## Results

Our Random Forest Classifer had a slight performance advantage over our basline model with an F1 score of 0.5851 and an Accuracy Score of 0.8896. Still, there was much room for improvement. 

Our Naive Bayes Classifier performed even better. After some hyperparameter tuning, we were able to achieve an F1 Score of 0.61 and an Accuracy Score of 0.88. 

When we used the Naive Bayes model on our manually-labeled dataset, we were pleasantly surprised with the results. The model performed extremely well with an F1 Score of 0.85 and an Accuracy Score of 0.949. 

## Conclusion

Our best model was the multinomial naive bayes classifier which got a cross validated F1 score around .61 and an accuracy score consistently over .88. This means that our model was 8% more accurate than guessing the dominant class and had a reasonable ability to distinguish between the two classes. From this we can say it would be a useful model to further narrow your search when trying to determine between events and non events, but not the end of the process. 

On the other hand, our manually-labeled dataset, however, leads us to believe that the model performs relatively well on data that it had a higher level of exposure to in training. This led us to believe that, with more data to train the model on, it is possible that this model could be used without much work following its implementation. 

### Next Steps

#### 1. More Data

- 11,000 is not a small data set but NLP learning always does better with more entries to learn from. This dataset could easily be 100,000 tweets

#### 2. Focus on Specific Keywords

- We saw in our manually-labeled dataset that our model performed very well. It would be interesting to do the analysis all over again with 10 or 20 keywords in mind instead of almost 220. The large number of keywords, we believe, made it harder for the model to learn the training data well. If we were to keep all 219 moving forward, we would need A LOT more tweets for each keyword. 

#### 3. Use Pretraied Vectors 

- We wanted to implement some pretrained vectors with GLoVe from Standford but unfortunately there model is seemingly facing some legal trouble right now. Their vectors were trained by billions of tweets.
- We expiremented with Word to Vec but the vectors learned from our data set did not improve any of our models. This code was messy and not in the final notebook. 

#### 4. Nueral Net

- A nueral net did not seem worth additional effort with our data set, but if we scraped more tweets and got a sufficiently large data set it could provide more classification power. 

## For More Information

See the full analysis in the [Jupyter Notebook](./Final_notebook.ipynb) or review this [presentation](./event_tweets.pdf).

For additional info, contact Aidan Coco or Michael Wirtz at
[aidancoco@gmail.com](mailto:aidancoco@gmail.com) and [michaelwirtz88@gmail.com](mailto:michaelwirtz88@gmail.com), respectively.

## Repository Structure
<pre>
├── Final_notebook.ipynb
├── README.md
├── data
│   ├── further_validation_tweets_target.csv
│   └── tweets.csv
└── images
    ├── bigram_negative_wordcloud_graph.png
    ├── bigram_positive_wordcloud_graph.png
    ├── class_imbalance_graph.png
    ├── countries_graph.png
    ├── event_keywords_graph.png
    ├── headline_image.png
    ├── hyperlink_graph.png
    ├── non_event_keywords_graph.png
    ├── subjectivity_graph.png
    ├── tweet_length_graph.png
    ├── unigram_negative_wordcloud_graph.png
    └── unigram_positive_wordcloud_graph.png </pre>