There is a lot of misconception between statistical model and machine learning model. In an attempt to understand the difference, I decided to use the the wine data to generate a statistical models as Linear Regression.
The main differences are:

#### 1. Aim of building the model

The ML Linear Regression model is designed to make the most accurate prediction possible of y. While the statistical Linear Regression model is designed for inference about the relationships between y and x.
Statistical models are mainly used for research purposes, while ML models are optimal for implementation in the production environment.

#### 2. Size of Data used to build the model

Machine learning does really well with a wide (high number of attributes) and deep (high number of observations). However statistical modeling is generally applied for smaller data with fewer attributes or they end up overfitting.


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

# Linear Regression Assumptions ☁️

In order to have a robust model, linear regression requires certain assumptions. Some assumptions are related to the relationship between the x & y and others are concerned with the Residuals/error terms. I will start defining the assumption and the reason behind it. 

## Assumption 1: Linear Relationship between y and x 📈

Since Linear regression describes a linear relationship between the dependent and independent variable/s; thus a linear relationship should exist between them.
It can be:
y = ax + b 
y = a log(x)

However, it can't be:
y = ax2x3 + b


we use a scatter_matrix; equivalent to pairplot; to check the relation of independent variables with the citric acid.

---------------------------


        fig = go.Figure(data=go.Splom(
                dimensions=[dict(label='fixed acidity',
                                 values=df_wine['fixed acidity']),
                            dict(label='volatile acidity',
                                 values=df_wine['volatile acidity']),
                            dict(label='free sulfur dioxide',
                                 values=df_wine['free sulfur dioxide']),
                            dict(label='total sulfur dioxide',
                                 values=df_wine['total sulfur dioxide']),
                            dict(label='chlorides',
                                 values=df_wine['chlorides']),
                            dict(label='alcohol',
                                 values=df_wine['alcohol']),
                            dict(label='residual sugar',
                                 values=df_wine['residual sugar']),
                            dict(label='citric acid',
                                 values=df_wine['citric acid'])],
                diagonal_visible=False, # remove plots on diagonal
                marker=dict(color=colors[2],
                            showscale=False, 
                            size=4,# colors encode categorical variables
                            line_color='white', line_width=0)
                ))

        fig.update_layout(
            template="plotly_white",
            title='Scatter Matrix',
            title_x=0.5,
            width=1200,
            height=1100,
            font=dict(size=13)
        )
        fig.show()
![newplot (1)](https://user-images.githubusercontent.com/81252980/208654618-058929d1-a15b-48de-866d-c8f411098c75.png)

We noticed that some variables are not in a linear relationship with the dependent variable. Then visualize them to make sure
![newplot (2)](https://user-images.githubusercontent.com/81252980/208654871-749e851f-09d7-45e1-9a6b-fe610169226d.png)


By looking at the plots we can see that `total sulfur dioxide` form somehow a linear shape with the `citric acid`, although some outliers exist. However, `free sulfur dioxide` and `alcohol` don't show any linearity with `citric acid`.Then we should eliminate these two variables from the model.

## Tune the model 

        # prepare the X for the model
        X = df_wine[['fixed acidity', 'volatile acidity', 'residual sugar',
               'chlorides', 'total sulfur dioxide']]
        Y = df_wine['citric acid']

        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.7,random_state=42)
        x_train_new = sma.add_constant(x_train)
        x_test_new = sma.add_constant(x_test)
        full_model = sma.OLS(y_train, x_train_new)
        res = full_model.fit()
        print(res.summary())

![Screenshot from 2022-12-20 12-20-09](https://user-images.githubusercontent.com/81252980/208655046-2762704d-526a-4e9b-af52-4f468667130e.png)


    print('Variance Inflating Factor')
    cnames = x_train.columns
    for i in np.arange(0,len(cnames)):
      xvars = list(cnames)
      yvar = xvars.pop(i)
      mod = sma.OLS(x_train[yvar], sma.add_constant(x_train_new[xvars]))
      res_1 = mod.fit()
      vif = 1 / (1- res_1.rsquared)
      print('--',yvar,'=',round(vif,3))

![Screenshot from 2022-12-20 12-20-38](https://user-images.githubusercontent.com/81252980/208655137-5b9b17d1-41af-43bb-abe3-847f57018bcf.png)

## Assumption 2: Check for Homoscedasticity 📊
Homoscedasticity means that the residuals have equal or almost equal variance across the regression line. By plotting the residuals against the predicted terms we can check the presence of any pattern.

------------



### **Graphical Method**


We plot the residuals against the predicted values or the X. If there is a definite pattern (like linear or quadratic or funnel shaped) obtained from the scatter plot then heteroscedasticity is present.


![newplot (3)](https://user-images.githubusercontent.com/81252980/208655274-20e71ae3-dfd3-4af0-9dad-2bd689fa4e4c.png)

## Assumption 2: Check for Homoscedasticity 📊
Homoscedasticity means that the residuals have equal or almost equal variance across the regression line. By plotting the residuals against the predicted terms we can check the presence of any pattern.

------------



### **Graphical Method**


We plot the residuals against the predicted values or the X. If there is a definite pattern (like linear or quadratic or funnel shaped) obtained from the scatter plot then heteroscedasticity is present.

we can see from the graph that there is no pattern/ shape presenting the residuals. Although, there are some values that are very far from the zero line. So, the assumption here is not violated 






### **Statistical Tests**


  




**Goldfeld Quandt Test:**

$$\mathcal{H}_{0}:  Residuals\ are\ homoscedastic\ $$ 

$$\mathcal{H}_{1}:  Residuals\ are\ not\ homoscedastic\ $$

        name = ['F statistic', 'p-value']
        goldfeld = sms.het_goldfeldquandt(res.resid, x_train_new)
        lzip(name, goldfeld)
        
[('F statistic', 0.9594617635104512), ('p-value', 0.6867050093000607)]


**Breusch Pagan Test for Heteroscedasticity**:

$$\mathcal{H}_{0}:  Residuals\ variances\ are\ equal\ (Homoscedasticity)$$ 

$$\mathcal{H}_{1}:  Residuals\ variances\ are\ not\ equal\ (Heteroscedasticity)\ $$ 


        import statsmodels.stats.api as sms
        from statsmodels.compat import lzip
        name = ['F statistic', 'p-value']
        test = sms.het_breuschpagan(res.resid, x_train_new)
        lzip(name, test)
        
[('F statistic', 48.06161324596686), ('p-value', 3.450531269363563e-09)]

In both test the p-value is more than 0.05 in Goldfeld Quandt Test and Breush Pagan Test, we accept their null hypothesis that error terms are homoscedastic.  ✅


## Assumption 3: Check for Normality of residuals 🧮
This assumptions requires the residual terms to be normally distributed


----------------

### **Graphically**
![newplot (4)](https://user-images.githubusercontent.com/81252980/208655631-e06b9a08-38c4-4160-8994-ec1624b564ee.png)
![newplot (5)](https://user-images.githubusercontent.com/81252980/208655634-cc6793ea-1eda-42b6-bcc2-a3fffe345bcd.png)

## Assumption 3: Check for Normality of residuals 🧮
This assumptions requires the residual terms to be normally distributed


----------------

### **Graphically**

### **Statistical Tests**

**Anderson Darling Test for checking Normality of Errors:**

$$\mathcal{H}_{0}:  The\ residuals\ follows\ a\ specified\ distribution $$ 

$$\mathcal{H}_{1}:  The\ residuals\ doesn't\ follows\ a\ specified\ distribution $$ 


        anderson_results = stats.anderson(res.resid, dist='norm')
        name = ['Overall p-value', 'p-value']
        lzip(name,anderson_results)

[('Overall p-value', 3.0803763706908285),

 ('p-value', array([0.574, 0.654, 0.784, 0.915, 1.088]))] 
 
- The distribution plot shows somehow a bell shape, skewed to the right a little bit. But acceptable. ✅

- The Q-Q plot shows that most values are present on straight line. ✅

- In the test the p-value is more than 0.05 then, we accept the null hypothesis that residuals follows a normal distribution. ✅


## Assumption 4: Dropping Multicollinear Variables 🔻

In regression, multicollinearity refers to the extent to which independent variables are correlated. Multicollinearity affects the coefficients and p-values, but it does not influence the predictions, precision of the predictions, and the goodness-of-fit statistics. If your primary goal is to make predictions, and you don’t need to understand the role of each independent variable, you don’t need to reduce severe multicollinearity

-------------------------

![Screenshot from 2022-12-20 12-24-50](https://user-images.githubusercontent.com/81252980/208655888-26d4aee6-62b7-4142-8dc2-37bb58abda65.png)
![newplot (6)](https://user-images.githubusercontent.com/81252980/208655912-d6d41085-1e8a-4756-b3b8-88b6c252b2bb.png)


VIF showed no multicollinearity. ✅

Heatmap shared the same result. ✅

## Assumption 5: No Autocorrelation of Residuals 🔗

Linear regression model assumes that error terms are independent. This means that the error term of one observation is not influenced by the error term of another observation. In case it is not so, it is termed as autocorrelation.


**Durbin Watson test is used to check for autocorrelation:**

$$\mathcal{H}_{0}:  Autocorrelation\ is\ absent\  $$ 

$$\mathcal{H}_{1}:  Autocorrelation\ is\ present\ $$ 

        from statsmodels.stats.stattools import durbin_watson
        durbin_watson(res.resid)

The value of the statistic will lie between 0 to 4. A value between 1.8 and 2.2 indicates no autocorrelation. A value less than 1.8 indicates positive autocorrelation and a value greater than 2.2 indicates negative autocorrelation

- Durbin test indicates no autocorrelation. ✅ 


# Final Evaluation 📍

        print(res.summary2())
![Screenshot from 2022-12-20 12-27-19](https://user-images.githubusercontent.com/81252980/208656378-cb5b660f-c79e-4f93-8f4b-6654b234a765.png)
        

## Overall Model Accuarcy:

This is evaluated by R-squared. R2 = 0.65 or 65%. Thus, our model **MAY BE** good enough to deploy on unseen data.


## Model Significance

Our Model:

$$\mathcal{Y}_{citric acid}= 0.075 {x}_{fixed acidity} +  0.0008 {x}_{free sulfur dioxide} - 0.33  $$

In order to prove that our linear model is statistically significant, we have to perform hypothesis testing for every β. Let us asume that:
$$\mathcal{H}_{0}: β_{1} = 0 $$
$$\mathcal{H}_{1}: β_{1} ≠ 0 $$
Simply, if β1 = 0 then the model shows no association between both variables 
$$\mathcal{Y}_{}= β_{0} + ε $$





To test the coefficient’s null hypothesis we will be using the t statistic. Look at the P>| t | column. These are the p-values for the t-test. In short, if they are less than the desired significance (commonly .05), you reject the null hypothesis. Otherwise, you fail to reject the null and therefore should toss out that independent variable.

Above, assuming a significance value of 0.05, our P-Value of 0.000 is much lower than a significance. Therefore, we reject the null hypothesis that the coefficient is equal to 0 and conclude that` fixed acidity` and `free sulfur dioxide` is an important independent variable to utilize.

Now, going back to the assumptions of the linear regression, some assumptions were violated. It seems that the free sulfur dioxide is skewing the results. 
Notice the t-score of both variables.
Generally, any t-value greater than +2 or less than – 2 is acceptable. We know that the higher the t-value, the greater the confidence we have in the coefficient as a predictor.Low t-values are indications of low reliability of the predictive power of that coefficient.

- fixed acidity           36.208    very high 🟢
- free sulfur dioxide     2.314     very low 🔴

Therefore in the model update i will remove the free sulfur dioxide from the equation.



