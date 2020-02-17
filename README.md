# Generate-Handwritten-Digits-Kernel-Density-Estimation

Kernel Density Estimation (KDE) is a non-parametric method for estimating the probability of a given point, using a dataset. This is an implementation of using numpy.

## Usage:

### Extended Univariate Gaussian Kernel
We introduce norm to leverage univariate kernel to work with high dimension (read the Report.md)

```python
# Choice of bandwidth
bandwidth = 2000

# Create KDE obj and fit the data
Estimator = KDE(kernel="univariate_gaussian",univariate_bandwidth=bandwidth)
Estimator.fit(x_train)

# Estimating bandwidth
est_density = Estimator.predict(x_test,y_test,batch_size=100)

```

### Multivariate Gaussian Kernel
Also, we can use multivariate gaussian kernel to estimate density at one point in high dimension

```python
# Bandwidth estimator choice
bandwidth_estimator="silverman"

train_size = 20000
test_size = 10

# Use multivariate kernel, and scott rule to estimate bandwidth H
Estimator = KDE(kernel="multivariate_gaussian", bandwidth_estimator=bandwidth_estimator)
Estimator.fit(x_train,y_train)

# Estimating bandwidth
est_density = Estimator.predict(x_test,y_test,batch_size=100)
```

**Note:**
- The bigger batch_size, the faster the operation goes but more ram-dependent
- If run out ram, please kindly decrease batch_size
