# A Vectorized Implementation of Multiple Linear, LASSO, and Ridge Regression Optimized Using Gradient Descent
## Program Information
<!---
Add information about how program is structured and how to run it here
-->
## Mathematics Behind the Models
For multiple linear, LASSO, and ridge regression the aim is to construct a model which yields numerical outputs that minimize an error metric (typically sum of squared errors or mean squared error). The model can be defined as below:  
![equation](https://latex.codecogs.com/png.latex?%5Cinline%20%24%5Cmathbf%7B%5Chat%7By%7D%7D%20%3D%20%5Cmathbf%7Bb%7D%20&plus;%20%5Cmathbf%7BX%7D%24)  
As we move along, let us clarify the notation. Boldface capital symbols denote matrices, boldface lowercase symbolds denote vectors, and normal text lowercase symbols denote scalars.
### Multiple Linear Regression
Firstly, let us take note of how the coefficients (the b vector) will be calculated for multiple linear regression. For our cost function, we will use mean squared error, or mse. More precisely, we will use half mse, as this will make derivation calculations simpler. The equation for this metric is given below:  
c = ![equation](https://latex.codecogs.com/png.latex?%5Cinline%20%24%5Cfrac%7B1%7D%7B2m%7D%28%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Cmathbf%7Bb%7D%29%5E2%24)  
So, our objective function is:  
![equation](https://latex.codecogs.com/png.latex?%5Cinline%20%24%5Cmin%20%5Cfrac%7B1%7D%7B2m%7D%28%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D%20-%202%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bb%7D%20&plus;%20%5Cmathbf%7Bb%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bb%7D%29%24)  
We can note that this is a convex function and therefore use gradient descent to find the global minimum of the function. Let us take the derivative of the cost function with respect to (w.r.t.) the coefficient vector, b, in order to perform gradient descent which will then minimize it:  
<img src="https://latex.codecogs.com/png.latex?%5Cinline%20%24%5Cfrac%7B%5Cpartial%20c%7D%7B%5Cpartial%7B%5Cmathbf%7Bb%7D%7D%7D%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%28-%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%20&plus;%20%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bb%7D%29%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Cmathbf%7BX%7D%5ET%28%5Cmathbf%7BX%7D%5Cmathbf%7Bb%7D%20-%20%5Cmathbf%7By%7D%29%24">  
The update for the b vector, with alpha as the learning rate, can be calculated as follows:  
<img src="https://latex.codecogs.com/png.latex?%5Cinline%20%24%5Cmathbf%7Bb%27%7D%20%3D%20%5Cmathbf%7Bb%7D%20-%20%5Cfrac%7B%5Calpha%7D%7Bm%7D%5Cmathbf%7BX%7D%5ET%28%5Cmathbf%7BX%7D%5Cmathbf%7Bb%7D%20-%20%5Cmathbf%7By%7D%29%24">
### Ridge Regression
For ridge regression, our cost function changes slightly from the case of multiple linear regression. There is now an L2 penalty term added. Let us take a look at the cost function now:  
<img src="https://latex.codecogs.com/png.latex?%5Cinline%20%24c%20%3D%20%5Cfrac%7B1%7D%7B2m%7D%28%28%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D%20-%202%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bb%7D%20&plus;%20%5Cmathbf%7Bb%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bb%7D%29%20&plus;%20%5Clambda%5Cmathbf%7Bb%7D%5ET%5Cmathbf%7Bb%7D%29%24">  
Lambda is the regularization parameter and is greater than or equal to 0 and less than or equal to 1. As this value increases, so does the cost; this means that the optimization will be incentivized to keep the values of the b vector low. This cost function is also convex and fully differentiable, so we can use gradient descent for optimization. We now take the derivative of the cost function w.r.t. the b vector in order to then find our update computation.  
<img src="https://latex.codecogs.com/png.latex?%5Cinline%20%24%5Cfrac%7B%5Cpartial%20c%7D%7B%5Cpartial%7B%5Cmathbf%7Bb%7D%7D%7D%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%28-%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%20&plus;%20%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bb%7D%20&plus;%20%5Clambda%5Cmathbf%7Bb%7D%29%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%28%5Cmathbf%7BX%7D%5ET%28%5Cmathbf%7BX%7D%5Cmathbf%7Bb%7D%20-%20%5Cmathbf%7By%7D%29%20&plus;%20%5Clambda%5Cmathbf%7Bb%7D%29%24">  
The update for the b vector is computed as follows:  
<img src="https://latex.codecogs.com/png.latex?%5Cinline%20%24%5Cmathbf%7Bb%7D%27%20%3D%20%5Cmathbf%7Bb%7D%20-%20%5Cfrac%7B%5Calpha%7D%7Bm%7D%28%5Cmathbf%7BX%7D%5ET%28%5Cmathbf%7BX%7D%5Cmathbf%7Bb%7D%20-%20%5Cmathbf%7By%7D%29%20&plus;%20%5Clambda%5Cmathbf%7Bb%7D%29%24">
### LASSO Regression
For LASSO regression, the cost function changes once again. We now 
### Why LASSO Regression Naturally Performs Feature Selection Whereas Ridge Regression Does Not
