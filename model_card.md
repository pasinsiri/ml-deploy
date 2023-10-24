# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A Gradient Boosting Classifier model aiming to classify the salary of a person based on his/her census data,
e.g. age, workclass, gender, native country, capital gain and loss. 
The model is trained with data from the UCI Census Income data

## Intended Use
The dataset that is used to train this model was created for educational purposes.

## Training Data and Evaluation Data
The dataset contains 32,561 rows of samples and 15 columns including *salary* which is the target column _(therefore the dataset contains 14 features)_. Features can be separated into two groups, demographic features, e.g. age, gender, and native-country, and educational and career features, e.g. working hours per week, education, capital gain, and capital loss.
In training the model we implemented a 5-Fold cross validation with the entire dataset.

## Metrics
We used F1 score as our metric. In the cross validation process, we compute the average F1 score from all the splits to represent the overall performance of the model. From the cross-validation process, the average F1 score is 0.66 with the minimum score being 0.63 and the maximum one being 0.68.

## Ethical Considerations
Features that have impact on a person's salary is not limited to the ones given in the data. There should be other features that can influence how much a person would be paid.

## Caveats and Recommendations
One can perform hyperparameter tuning in order to find the best set of hyperparameters of the model, or implement different machine learning models, e.g. Random Forest, or Light Gradient Boosting Machine, to this dataset and see whether the other models give more accurate result. 