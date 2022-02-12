There is a lot of misconception between statistical model and machine learning model. In an attempt to understand the difference, I decided to use the same wine data to generate both models as Linear Regression and check the difference between them. 

I will pinpoint the main differences between the two models, then elaborate on each model by itself.


1- Origin

Statistical models began evolving in the 18th century in response to the novel needs of industrializing sovereign states. It started with collecting information about the states, particularly demographics such as population. Later, this extended to include all collections of information of all types including the analysis and interpretation of such data.

Machine learning is a very recent development. It came into existence in the 1990s as steady advances in digitization and cheap computing power enabled data scientists to stop building finished models and instead train computers to do so. 



3- Type of Data
Machine learning does really well with a wide (high number of attributes) and deep (high number of observations). However statistical modeling is generally applied for smaller data with fewer attributes or they end up overfitting.



5- Aim
The ML Linear Regression model is designed to make the most accurate prediction possible of y. While the statistical Linear Regression model is designed for inference about the relationships between y and x.
Statistical models are mainly used for research purposes, while ML models are optimal for implementation in the production environment.


6- Splitting of Data
In Statistical Linear regression, the data is split between 70% training and 30% testing. Note here, although we say training, we don't mean that actual training happens

The training data is used to develop the model and the testing is used to test the model.
In ML, the data is split into 50%,25%, and 25%, training, testing, and validation respectively.

Model is developed on training, and hyperparameters are tuned on validation data and finally get evaluated on test data.


