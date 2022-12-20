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
![Screenshot from 2022-12-20 11-55-21](https://user-images.githubusercontent.com/81252980/208650977-f7b9f550-c6af-4128-a6ee-48030a418853.png)

    df_wine.columns
![Screenshot from 2022-12-20 11-55-35](https://user-images.githubusercontent.com/81252980/208650989-d3549f23-5821-405c-b37c-31b843c5df37.png)

    # no null values are detected
    df_wine.isnull().sum()

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

