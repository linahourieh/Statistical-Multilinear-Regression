There is a lot of misconception between statistical model and machine learning model. In an attempt to understand the difference, I decided to use the same wine data to generate both models as Linear Regression and check the difference between them. 

I will pinpoint the main differences between the two models, then elaborate on each model by itself.

3- Type of Data
Machine learning does really well with a wide (high number of attributes) and deep (high number of observations). However statistical modeling is generally applied for smaller data with fewer attributes or they end up overfitting.


5- Aim
The ML Linear Regression model is designed to make the most accurate prediction possible of y. While the statistical Linear Regression model is designed for inference about the relationships between y and x.
Statistical models are mainly used for research purposes, while ML models are optimal for implementation in the production environment.


# Loading Essential Libraries 📚

we start first by importing essential libraries

    # for data manipulation
    import pandas as pd
    import numpy as np


    # for vizualization
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # for statistical tests
    import statsmodels.api as sma
    import statsmodels.stats.api as sms
    from statsmodels.compat import lzip
    from statsmodels.tools.tools import add_constant
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.graphics.gofplots import qqplot
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score 

    import scipy.stats as stats


# Reading Data Set 👓

    url = 'https://raw.githubusercontent.com/linahourieh/Wine_Quality_Multilinear_Reg/main/winequality-red.csv'
    df_wine = pd.read_csv(url)
    df_wine.head()
![Screenshot from 2022-12-20 11-55-21](https://user-images.githubusercontent.com/81252980/208650868-5a4b47df-9f05-415e-bbb1-41882e3b8b8c.png)


    df_wine.shape
![Screenshot from 2022-12-20 11-55-35](https://user-images.githubusercontent.com/81252980/208650968-3726e657-03f9-48c4-ad78-1ad3c5f019c2.png)

    df_wine.describe()
![Screenshot from 2022-12-20 11-55-43](https://user-images.githubusercontent.com/81252980/208651282-0789d07e-3cb8-4f36-a3c9-4d4df2d5c295.png)


    df_wine.columns
![Screenshot from 2022-12-20 11-55-52](https://user-images.githubusercontent.com/81252980/208651250-43fb60de-7172-4f31-9977-b98590ee92c6.png)

    # no null values are detected
    df_wine.isnull().sum()
![Screenshot from 2022-12-20 11-59-49](https://user-images.githubusercontent.com/81252980/208651478-e024d32d-02ee-4e93-8f5f-cb0f4bc0093f.png)

    # define some colors
    colors = ['#2B463C', '#688F4E', '#B1D182', '#F4F1E9']

    # function to generate a continuous color palette from 2 colors
    def colorFader(c1,c2,mix=0):
        c1=np.array(mpl.colors.to_rgb(c1))
        c2=np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

    c1=colors[1] 
    c2=colors[3] 
    n=1000

    # list containing color series
    c=[]
    for x in range(n+1):
        c.append(colorFader(c1,c2,x/n))


# Model Development 🛠

### **Forward Selection**
When picking the independent variables to our model, we should rely more on common sense and our background knowledge.

Here is a good [article](https://quantifyinghealth.com/variables-to-include-in-regression/#:~:text=As%20a%20rule%20of%20thumb,sense%20and%20your%20background%20knowledge.) explaining how you should pick your independent variables.


### **Backward Selection**
Here we will follow this methodology. It is more common in industry.We include all variables in our model then, according to p-value and VIF, we eliminate variables accordingly.


### **Key Metrics**

> **AIC:**
- No absolute value is significant. It is a relative measure, the lower the better

> **Adjusted R-squared:**
- It is >= 0.7

> **Individual variable's P-value (P>|t|):**
- It is =<0.05

> **Individual variable's VIF:**
- It is <5



### Iterations Summary

| Number      | column eliminated | p-value | vif  | r-sqr |
| :---        |    :----:         |    ---: | ---: | ---: |
| 1           | density           |  0.633  |  6.4 | 0.68 |
| 2           | sulphates         | 0.292   | 1.37 | 0.68 |
| 3           | quality           | 0.21    | 1.51 | 0.68 |
| 4           | pH                | 0.225   |  2.1 | 0.68 |




###**Key Metrics**

> **AIC:**
- Reduced from -1736 from iteration -1740 to in iteration 5

> **Adjusted R-squared:**
- 0.679 --> 0.679

> **Individual variable's P-value (P>|t|):**
- It is =<0.05

> **Individual variable's VIF:**
- It is <5



# Testing
        # Prediction of Data
        y_pred = full_results.predict(x_test_new)
        y_pred_df = pd.DataFrame(y_pred)
        y_pred_df.columns = ['y_pred']

        pred_data = pd.DataFrame(y_pred_df['y_pred'])
        y_test_new = pd.DataFrame(y_test)
        y_test_new.reset_index(inplace=True)
        pred_data['y_test'] = pd.DataFrame(y_test_new['citric acid'])

        # R-Squared Calculation
        rsqd = r2_score(y_test_new['citric acid'].tolist(),
        y_pred_df['y_pred'].to_list())

        print("Training R-square value = ", round(full_results.rsquared_adj,4))
        print("Test R-square value = ", round(rsqd,4))
  
  
Training R-square value =  0.6788

Test R-square value =  0.6731


- The training and testing R-sqr are very similar to each other. 

- then, the relationship between the dependent and the independent variables could be represented by a linear Regression.

- We can say that these Independent variables explains 67% of the variablity in citric acid values.



