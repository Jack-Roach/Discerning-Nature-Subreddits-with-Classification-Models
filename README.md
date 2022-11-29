# **Nature vs. Nature: Discerning Nature Subreddits with Classification Models**
## By Jack Roach
#### Copy of my repository from my time in General Assembley's Data Science Immersive Course
### Background
Reddit is one of the largest and most recognized user-driven websites on the internet. It is home to thousands of communities known as subreddits (or simply 'subs') that thrive off of user-submitted posts. Content includes but is not limited to all types of news, visual media, questions, opinions, and discussion prompts. The focus of a subreddit can be as ambiguous as a [picture](https://www.reddit.com/r/pics/) to something as specific as a [cat standing up](https://www.reddit.com/r/CatsStandingUp/).

There are many subreddits within reddit that have overlapping content and themes that blend with those of other subreddits. Two such examples are r/NatureIsFuckingLit and r/NatureIsMetal. Here are the self-given descriptions of these two communities.

## [r/NatureIsFuckingLit](https://www.reddit.com/r/NatureIsFuckingLit/)
Subreddit Description: We are here to appreciate the awesome majesty and incredibly cool aspects of nature. 🔥

## [r/NatureIsMetal (NSFW)](https://www.reddit.com/r/natureismetal/)
#### **<u>Important Note</u>**: This subreddit contains graphic scenes that occur in nature. Please do NOT visit this subreddit if you are at all squeemish to naturally occuring animal violence. None of this subreddit's NSFW content will be described in any sort of detail throughout this data analysis project.
Subreddit Description: Badass pictures, gifs and videos of the awesome true brutality of nature.

As you may be able to infer from the titles and descriptions, these two communities share overlapping characteristics with distinct differences. Both subreddits are nature-oriented subs, share half a name, and are intended to evoke a feeling of awe towards nature.

### Problem Statement
I hypothesize that r/NatureIsFuckingLit and r/natureismetal are similar enough in theme that you would be hardpressed to consistently discern their posts without the photography and videos included within them.
My goal is to disprove this claim by using natural language processing to create a classification model that is capable of accurately predicting which aforementioned nature subreddit a post comes from using the contents of its title.

### Data Collection
#### [Modeling Data](data/edited_data.csv)

In order to gather subreddit post data, I utilized [Pushshift API](https://github.com/pushshift/api), a reddit web-scraping tool developed by reddit user u/stuck_in_the_matrix that is capable of gleaning post data from reddit. I gathered 5000 of the most recent posts from r/natureisfuckinglit and r/natureismetal that were posted before April 29th, 2022. From each post,I extracted the post title, body text, and the subreddit it was posted in. The majority of posts in both subreddits had empty body text values, which was expected due to both subreddits being image and video focused. With these 10,000 data entries, I can train a classification model to predict which subreddit a post was posted in and answer our problem statement.

A classification model with an accuracy score of 90% or greater will be considered sufficiently accurate.

### Exploratory Data Analysis
The first trend in the data was quick to show itself, with the majority of the posts collected from r/NatureIsFuckingLit containing one or more 🔥 emojis. 2817 out of the 5000 scraped posts from r/NatureIsFuckingLit contained at least one 🔥 emoji. This is due to rule #2 in the r/NatureIsFuckingLit sub, which requires users to put a 🔥 at the beginning of their post title. The r/natureismetal data, on the other hand, only contained 17 posts which contained a 🔥. In 13/17 of these posts, the 🔥 emoji appears at the beginning of the title which suggests that the user made the same post to r/NatureIsFuckingLit without bothering to edit the title.

Another thing I noticed was the presence of spam posts in both subreddits, varying from weight loss ads to crypto scams. They do not appear to take up a significant portion of the data set, so I chose not to remove any as I would not be able to go through all 10,000 posts by hand, and do not currently have the means to identify and prune spam autonomously. I am also unconcerned by potential spam because r/natureismetal and r/natureisfuckinglit have eight mutal moderators out of the 10 total moderators they have each. Spam is likely handled/detected the same way in both subreddits because of this, making any data abberations spam titles might cause consistent across both subreddits.

Lastly, I removed the one post in the dataset that contains body text, which turned out to be an off-topic post about a no-spoilers policy concerning the film, Spiderman: No Way Home that was released in December of 2021.

To see which words were most common in the dataset, I used CountVectorizer to vectorize the title words in the dataset. When vectorizing words by subreddit, the fire emoji was the most freqeunt "word" appearing in r/NatureIsFuckingLit followed by nature. "Eating" is the most frequently occuring word that appears in r/natureismetal post titles. Although ranked in differently, both r/natureismetal and r/NatureIsFuckingLit had the words 'nature', 'tree', 'oc', 'red', 'like', and 'baby' in their top 25 most frequent words. 'oc' refers to the '[OC]' tag used to indicate that a post contains original content.

#### In order to challenge myself for this model, I will not be including the fire emoji as a word in my training and testing data 

### Our Models
Accuracy indicates the percentage of posts in the test data that the model will correctly classify

|Model|Accuracy %|
|---|---|
|Logistic Regression with Count Vectorizer including the fire emojis|82.71%
|Baseline Model| 50.02%| 
|Grid Search on K Neighbors Classifier with Count Vectorizer|67.71%|
|Logistic Regression with Count Vectorizer|74.11%|
|Grid Search on Extremely Random Trees with Count Vectorizer|74.99%|
|Grid Search on Random Forest with Count Vectorizer|75.43%|
|Grid Search on Multinomial Naive Bayes with Tfidf Vectorizer|75.79%|
|Grid Search on Logistic Regression with Count Vectorizer|76.07%|
|Grid Search on Logistic Regression with Tfidf Vectorizer|76.19%|
|Grid Search on Multinomial Naive Bayes with Count Vectorizer|76.43%|
|Voting Classifier - Ensemble Model|76.75%|

### Production Model
I analyzed a variety of classification models in order to ensure the best chance of building an accurate predictive model. After a training variety of models, I decided to use a Voting Classifier to combine the three models with the best results thus far. I did not know if the Voting Classifier ensemble would be greater than the individual models processed through it, but after training it, it showed an accuracy slightly but consistently higher than the previous best model, the Multinomial Naive Bayes (using CV).

Out of all of our models that do not include the fire emoji data, the Ensemble Model performs the best out of the ones we've tested. The ensemble model is built from the Multinomial Naive Bayes w/CV model, the Logistic Regression w/Tfidf model, and the Extremely Random Trees w/CV model, all three of which are equally weighted in the soft voting classifier.
The model classifies 78% of r/natureismetal posts from the testing set correctly while it classifies 75% of r/NatureIsFuckingLit posts correctly.

We used 2499 posts to test the model, and the model correctly classified 1920 out of the 2499 posts.

There were 270 instances of the model falsely predicting a post from r/natureismetal as a r/NatureIsFuckingLit post.

There were 309 instances of the model false predicting a post from r/NatureIsFuckingLit as a r/natureismetal post.

The reason that the model misclassifies r/NatureIsFuckingLit posts more often than r/natureismetal posts is likely because you may find content suitable for r/natureismetal in the r/NatureIsFuckingLit subreddit such as a photo of fierce looking bear, but you won't find as many posts suitable for r/NatureIsFuckingLit in the r/natureismetal subreddit (e.g. a pretty landscape being posted in r/natureismetal).
### Conclusion
I was able to create a model that could correctly classify reddit posts from r/natureismetal and r/NatureIsFuckingLit with an accuracy of 76.75% based on their titles. While I was able to make a model substantially more accurate than the baseline model, a model with 76.76% accuracy is not that strong and falls below our stated accuracy goal. This score means that the model will only be able to classify a post correctly approximately three times out of four. From this we can infer that the characteristics of titles in both of these subreddits are similar enough to make classifying them a challenge, at least without taking advantage of r/NatureIsFuckingLit's 🔥 rule.

My recommendation is to implement an image recognition algorithm that can identify objects in photos and video frames in order to build a classification model that can consistently classify posts in these two subreddits. This will help to make subreddit posts more distinguishable in future data analysis projects.

### Future Analysis and ideas to improve the model
- Implement an algorithm that can detect objects in pictures
- Include all emojis in natural language processing
- Train a model that includes the 🔥 emoji as a unique word
- Test additional models such as the Ada Boost Classifier
- Test the Tfidf Vectorizer with all models
- Conduct project on a higher-end pc to faciliate grid searches with large amounts of parameter combinations. At one point I had intialized a paramater set that would have resulted in 4000+ combinations for grid search to check, something my computer would not handle very well.
- Test additional parameters with grid search, such as voting weights in the Vote Classifier model.
- Include other nature subreddits such as r/nature, r/nature**was**metal, r/natureporn (not actually nsfw), and r/earthporn to see how a model performs in classifying posts when there's more than two subreddits to choose from.