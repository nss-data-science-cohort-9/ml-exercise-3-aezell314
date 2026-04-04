## Machine Learning Exercise 3: Bias and Variance

**Bias** refers to the error introduced by approximating a complex real-world problem with a simplified model, while **variance** refers to the model's sensitivity to fluctuations in the training data. A linear regression model has high bias and low variance; it makes strong assumptions about the data (linearity) but is stable across different datasets. If these strong assumptions are not correct, there will be places where it systematically overestimates or underestimates. In contrast, a decision tree model has low bias and high variance;it can capture complex patterns but is prone to overfitting, especially if deep and unpruned. This means that it can start to memorize the training data rather than capturing patterns that generalize.

1. Fit a linear regression model to the housing data, using sqft_living to predict price. Check the mean squared error on the training data and the test data.

2. Repeat this but with a [DecisionTreeRegresor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html). Again check the mean squared error on the training data and the test data. How does what you see differ from the linear regression model?

One way of avoiding overfitting is by restricting the flexibility of the model. We can do this with a decision tree by restricting the number of splits that it can perform. 

3. Fit a DecisionTreeRegressor where you restrict the max_depth to 5. Again check the mean squared error on the training data and the test data. What do you notice now?

When working with machine learning models, we often have to balance bias and variance. This is called the [bias-variance tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff). One method of this is through [regularization](https://www.ibm.com/think/topics/regularization), where we restrict the complexity of the model, adding some bias but reducing the variance, which can lead to a lower mean squared error on the test set.

Lasso and ridge regression do this by adding a penalty term based on the size of the coefficients. Smaller coefficients means that the model has less flexibility. The neat thing about these types of models is that they determine how to allocate the coefficients automatically as part of the model fitting process, so we can start with a large set of potential predictors and allow the model fitting to determine which ones to focus on.

For the next part of the exercise, we'll see how we can add complexity to our model but control the complexity through regularization.

4. So far, we've only been predicting based off of the square footage of living space. Fit a new linear regression model using all variables besides id, date, price, and zipcode. How well does this model perform on the test set compared to the model with just square footage of living space?

5. Try fitting a lasso and ridge model. Becuase lasso and ridge have penalty terms based on the size of the coefficients, and the size of the coefficients depends on the scale of the variable, you'll want to scale the features first so that they are on comparable scales. Create a [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) object where the first step is applying a [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) and the second step is either a lasso or ridge model. Because these models have a hyperparameter controlling regularization strength, you'll want to use the [LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html) and [RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html) models, which will select values for the regularization strength using cross-validation.

You likely didn't see much difference between the regular linear regression model and the lasso or ridge model. Let's see what happens if we add more complexity through feature interactions. We can capture more complex relationships between the predictor variables and the target variable by multiplying the predictors together before fitting the model. For example, the interaction between sqft_living and bedrooms will let the model capture if the impact of square footage depends on the number of bedrooms.

6. Add [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) to your pipeline after the standard scaler. Try using degree 2 features. How does this change the performance of the regular linear regression model, the lasso model, and the ridge model? 

The lasso penalty tends to cause some coeffients to zero out, so it can be viewed as a method of automatic feature selection.

7. Look at the set of coefficients for the lasso model. What percentage of the coefficients are zero? What are the largest non-zero coefficients?

8. A new hyperparameter that we have is the degree of the polynomial we're using. So that we're not overfitting to the test set, we need to use cross-validation to select this value. Set up a [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to try out polynomial degrees from 1 to 3 and to try LinearRegression, LassoCV, and RidgeCV models. Use 'neg_mean_squared_error' as the error_score. Which combination does it find does the best? 

** If you've reached this point, let your instructors know so that they can check in with you. **

Stretch Goals:

1. With home prices, there are some extremely large values for price, which can overly-influence the mean squared error calculation. See what happens if you optimize for mean absolute error instead. Alternatively, try using a [TransformedTargetRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html) to predict the log of price instead of the raw price.

**Bonus Exercise** We've seen how a decision tree model has move flexibility, which mean higher variance compared to a linear regression model. One way of understanding variance is that variance describes how sensitive the model is to the training data. A model with high variance can produce vastly different predictions when trained on different subsets.

In this bonus exercise, you'll get to see this in action.

Generate 1000 different linear regression fits, where you only use sqft_living as the predictor variable. For each fit, choose a random sample from the full dataset of size 1000. 
Using the np.linspace function, generate a grid of 100 equally-spaced points between 500 and 3000 and generate predictions on those points. Plot the mean prediction, the 5th percentile, and the 95th percentile for the predictions from these thousand models. Repeat this for a decision tree model. Then do it for a decision tree model with a max_depth of 5.

How do these predictions differ? Which have the most variability?