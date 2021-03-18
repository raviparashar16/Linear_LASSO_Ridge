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
In the above equation, m is the length of the y vector. So, our objective function is:  
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
For LASSO regression, the cost function changes once again. We now add an L1 penalty to the original multiple linear regression cost function. The cost function is as follows:  
<img src="https://latex.codecogs.com/png.latex?%5Cinline%20%24c%20%3D%20%5Cfrac%7B1%7D%7B2m%7D%28%28%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D%20-%202%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bb%7D%20&plus;%20%5Cmathbf%7Bb%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bb%7D%29%20&plus;%20%5Clambda%7C%7C%5Cmathbf%7Bb%7D%7C%7C_1%29%24">  
Unfortunately, although this cost function is convex, it is not differentiable. Luckily, we can use some properties of derivatives and functions that will nevertheless allow us to use a gradient descent approach to minimize the cost function towards its global minimum. The part of this cost function which is the same as the cost function for multiple linear regression is differentiable, but the penalty is not. We can use the Moreau-Rockafeller theorem which shows that <img src="https://latex.codecogs.com/png.latex?%5Cinline%20%24%5Cpartial%20%5Bf%28x%29%20&plus;%20g%28x%29%5D%20%3D%20%5Cpartial%20f%28x%29%20&plus;%20%5Cpartial%20g%28x%29%24"> . This means that the derivative of the cost function will be the derivative of the multiple linear regression part plus the derivative of the penalty part. Let us take the derivative of the multiple linear regression portion with respect to the jth component of the b vector (this corresponds to the jth column of the X matrix): <img src="https://latex.codecogs.com/png.latex?%5Cinline%20%24%5Cfrac%7B1%7D%7Bm%7D%28%5Cmathbf%7BX%7D_j%5ET%28%5Cmathbf%7BX%7D_j%5Cmathbf%7Bb%7D_j%20-%20%5Cmathbf%7By%7D%29%29%24">  
Now let us take the derivative of the penalty part. Using subdifferentials, we obtain: <img src="https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cleft%5C%7B%20%5Cbegin%7Barray%7D%7Bll%7D%20-%5Clambda%20%26%20b_j%20%3C%200%20%5C%5C%20%5Csmall%5B-%5Clambda%2C%20%5Clambda%20%5Csmall%5D%20%26%20b_j%20%3D%200%20%5C%5C%20%5Clambda%20%26%20b_j%20%3E%200%20%5C%5C%20%5Cend%7Barray%7D%20%5Cright.">  
However, we know that convex equations have a minimum. The minimum of the penalty portion of the cost function occurs x=0, and even thought it is not differentiable at that point, the subgradient is 0. With what we know now, we can add the derivatives together to obtain the derivative of the whole cost function w.r.t. the jth component of the b vector.  
<img src="https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cfrac%7B%5Cpartial%20c%7D%7B%5Cpartial%7B%5Cmathbf%7Bb%7D_j%7D%7D%20%3D%20S%28%5Cmathbf%7BX%7D_j%2C%5Cmathbf%7Bb%7D_j%2C%5Cmathbf%7By%7D%2C%20%5Clambda%29%3D">
<img src="https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cleft%5C%7B%20%5Cbegin%7Barray%7D%7Bll%7D%20%5Cfrac%7B1%7D%7Bm%7D%5Cmathbf%7BX%7D_j%5ET%28%5Cmathbf%7BX%7D_j%5Cmathbf%7Bb%7D_j%20-%20%5Cmathbf%7By%7D%29%20-%20%5Clambda%20%26%20%5Cmathbf%7Bb%7D_j%20%3C%200%20%5C%5C%20%5Cfrac%7B1%7D%7Bm%7D%5Cmathbf%7BX%7D_j%5ET%28%5Cmathbf%7BX%7D_j%5Cmathbf%7Bb%7D_j%20-%20%5Cmathbf%7By%7D%29%20%26%20%5Cmathbf%7Bb%7D_j%20%3D%200%20%5C%5C%20%5Cfrac%7B1%7D%7Bm%7D%5Cmathbf%7BX%7D_j%5ET%28%5Cmathbf%7BX%7D_j%5Cmathbf%7Bb%7D_j%20-%20%5Cmathbf%7By%7D%29%20&plus;%20%5Clambda%20%26%20%5Cmathbf%7Bb%7D_j%20%3E%200%20%5C%5C%20%5Cend%7Barray%7D%20%5Cright.">  
Now that we have the derivative, we can calculate the update for the jth component of the b vector:  
<img src="https://latex.codecogs.com/png.latex?%5Cinline%20%5Cmathbf%7Bb%7D%27_j%20%3D%20%5Cmathbf%7Bb%7D_j%20-%20%5Calpha%20S%28%5Cmathbf%7BX%7D_j%2C%5Cmathbf%7Bb%7D_j%2C%5Clambda%2C%5Cmathbf%7By%7D%29">
### Why LASSO Regression Naturally Performs Feature Selection Whereas Ridge Regression Does Not
Let us now turn our attention to why LASSO regression is naturally able to perform feature selection and why ridge regression cannot.
