# DSML2023 - Team Tesla
# Training a model to compute french difficulty
*This is our way of resolving the coding challenge!*

## Team Tesla members: 
* Elias Amine Sam
* Hamza Khalif Warsame

## Project structure
* Introduction
* Part 1
* Part 2
* Best results
* Video presentation
* References

## IMPORTANT LINK
Link to download the folder with our app and all the models : https://drive.google.com/drive/folders/166fG4T8vKzYRqfC1kYXeBvDGm3nsJPo5?usp=drive_link
Please put the following folders in the same folder as the ClassificationApp.py : LogisiticRegressionModel, RandomForestModel and FlaubertModel

## Introduction
In the context of our Data Science and Machine Learning project, we were tasked with developing a classifier capable of determining the difficulty level of French texts, categorized into levels A1, A2, B1, B2, C1, and C2. Our approach revolved around two distinct datasets. The first, a training set, included 4800 texts with their associated difficulty levels. From this set, we developed a predictive model which was then tested on a second set of 1200 texts.

![Methodological approach](https://github.com/Lyes14/DSML2023_Tesla/blob/main/Images%20(for%20README)/Methodolgy.png?raw=true)


Our process unfolded in two parts :

For the first part, after an exploratory data analysis, we explored four different models: Logistic Regression, K-Nearest Neighbors, Decision Tree, and Random Forest. These models were applied directly to our training data, without any cleaning data. Moreover, a data augmentation strategy was implemented with the aim of refining and improving our models' performance.
The relevance of each model was assessed using various metrics: precision, recall, F1 score, and accuracy. These indicators allowed us to gauge the effectiveness of our models in classifying texts.

In the second part, after consulting with a Ph.D. student who recommended the usage of text embeddings with pre-trained models, we used CamemBERT (BERT) but since we learned that FlauBERT (another pre-trained model for french texts) can give us some better results in our context, we then chose this model. Finally, we performed fine-tuning on it and evaluate its performance with multiple metrics.

## Part 1
We initially tried an approach based on basic models which are logistic regression, kNN, decision tree, and random forest (see notebook : BasicModels.ipynb) to classify the difficulty of sentences according to their level. After using them, we wanted to evaluate them with accuracy, precision, F1-score, and recall, and obtained the following results:



| Model          | Accuracy | Precision | F1-Score | Recall |
|-----------------|----------|-----------|----------|--------|
| Logistic Regression. | 0.4667     | 0.4656      | 0.4640     | 0.4667   |
| kNN             | 0.3146     | 0.3721      | 0.2947     | 0.3146   |
| Decision Tree  | 0.2958     | 0.2897      | 0.2958     | 0.2907   |
| Random Forest | 0.3708     | 0.3857      | 0.3537     | 0.3708   |

Following this, we opted for a features augmentation (see notebook : BaicsModelsWithFeaturesAugmentation.ipynb), which would allow us to enrich our training set. We tried this and obtained the following results:

| Model          | Accuracy | Precision | F1-Score | Recall |
|-----------------|----------|-----------|----------|--------|
| Logistic Regression. | 0.4167     | 0.4255      | 0.4108     | 0.4167   |
| kNN             | 0.3135     | 0.3055      | 0.3020    | 0.3135   |
| Decision Tree  | 0.3219     | 0.3211      | 0.3212     | 0.3219   |
| Random Forest | 0.4385     | 0.4330      | 0.4385     | 0.4306   |



Here we see that even with feautres augmentation, there was no improvement in our model, which is further evidenced by the confusion matrix below for the random forest, for example. We observe that it predicts sentences of level A1 quite well but still confuses the level of some sentences. Therefore, we can continue to explore in an effort to find a model that predicts better.

![Confusion matrix - Random Forest](https://github.com/Lyes14/DSML2023_Tesla/blob/main/Images%20(for%20README)/RandomForestConfusionMatrix.png?raw=true)

Another approach we considered was to augment our data by attempting to create new sentences based on the original phrases in the dataset, either by rephrasing them or simply by replacing certain words with synonyms. However, after evaluation, we concluded that this method was likely to lead to overfitting. This was evident when we assessed the accuracy on the test set of our training dataset (20%), where we indeed observed significantly higher accuracies, such as 0.65 and 0.69. But when we submitted our predictions on Kaggle, the accuracy was markedly lower. This discrepancy indicated that while our model performed well on the training data, it failed to generalize effectively to new, unseen data.

In our project, we explored the use of text embedding techniques, which are essential for understanding and processing natural language. Text embeddings transform words and phrases into vectors of numbers, allowing machine learning models to process text more effectively.

Following discussions with assistants and pre-doctoral researchers, we focused our attention on specific text embedding models, namely Camembert and Flaubert, known for their effectiveness with French text. We initially tested the Camembert model, which is based on the BERT architecture and specially trained on French data. Camembert proved useful for capturing linguistic context and the semantics of phrases.

However, in pursuit of even better results, we ultimately decided to turn to the Flaubert model. Flaubert, also based on the BERT architecture and optimized for French, offers potentially more advanced language comprehension capabilities. We considered that Flaubert might be better suited to our specific needs, particularly due to its ability to grasp subtle nuances in text, which is crucial for our goal of determining the difficulty of sentences.



## Part 2
We therefore loaded the FlauBERT model (flaubert/flaubert_based_cased), we split our data into training and test sets. While this division is crucial for assessing the model's generalization, it might slightly lower the accuracy on the test set compared to the results on Kaggle, as the model is only trained on a portion of the data. Following this, we prepared the data for training, followed by an essential phase of fine-tuning to determine the optimal parameters. This process involved the meticulous adjustment of several key hyperparameters to optimize the model's performance.

1. Number of Epochs: We adjusted the number of epochs, which is crucial for the model's learning. An epoch corresponds to a complete cycle of processing the entire training dataset. Increasing the number of epochs provides the model with more opportunities to learn from the data, but too high a number may lead to overfitting. After experimentation, we opted for a total of 8 epochs, a number that seems balanced for your model.

2. Learning Rate: The learning rate is another vital parameter. It controls the speed at which the model adjusts its parameters in response to errors. A too high rate can lead to premature and suboptimal convergence, while a too low rate can excessively slow down the learning. We found that a learning rate of 5e-5 improved your results.

3. Batchsize : In addition to these settings, we also set the batch size to 8, contributing to better memory management and increased learning efficiency. The adjustment of these parameters, potentially accompanied by other adjustments such as regularization, led to a significant improvement in the performance of our model.

The results after this fine-tuning proved to be significantly superior to those of the baseline models, with or without data augmentation, indicating that the adjustments were well-suited to the specifics of your task of predicting sentence difficulty.

| Model          | Accuracy | Precision | F1-Score | Recall |
|-----------------|----------|-----------|----------|--------|
| Flaubert | 0.5812    | 0.5891      | 0.5797     | 0.5819   |

Among the various hyperparameter combinations we experimented with, we tried several learning rates: 5e-5, 4e-5, 3e-5, and 1e-5. Additionally, we tested two different batch sizes: 16 and 8, and also experimented with a range of epochs, from 3 to 10.
These tests consistently yielded results with an accuracy exceeding 0.57.

Moreover, we see here that through our confusion matrix, the labels are better predicted. By observing the diagonal, we can assert that the accuracy is higher in this model and we notice that it confuses the classes less than the previous model.

![Confusion matrix - Flaubert](https://github.com/Lyes14/DSML2023_Tesla/blob/main/Images%20(for%20README)/FlaubertConfusionMatrix.png?raw=true)

## Best results
The model with the aforementioned hyperparameters (lr=5e-5, batch_size=8, epochs=8)  was subsequently retrained on the entire training dataset. The predictions from this model enabled us to achieve our best accuracy: **0.62**. This accomplishment helped us secure the second place on the leaderboard. However, there are still avenues for improvement that we identified. For instance, we could have retrained multiple models (such as CamemBERT-based and CamemBERT_large) and generated predictions from them. This would have allowed us to aggregate multiple predictions, enabling us to select a final prediction for each sentence based on the majority vote. Additionally, exploring other models like LAMA or GPT could have potentially yielded even better results.

## Video Presentation:
Link to the video : 
https://www.youtube.com/watch?v=1zxl1Rc-iWQ&ab_channel=Wider

## References
### Main references: 
* https://scikit-learn.org/stable/index.html
* https://github.com/michalis0/DataMining_and_MachineLearning
### Other references:
* https://docs.python.org/3/howto/regex.html
* https://towardsdatascience.com/ensemble-methods-or-democracy-for-ai-bac2fa129f61
* https://datacorner.fr/nltk/
* https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769
* https://goodboychan.github.io/python/datacamp/natural_language_processing/2020/07/15/01-Regular-expressions-and-word-tokenization.html#Introduction-to-regular-expressions
* https://github.com/madhurimamandal/Text-classification-into-difficulty-levels

