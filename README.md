# Generate-Handwritten-Digits-Kernel-Density-Estimation

Kernel Density Estimation (KDE) is a non-parametric method for estimating the probability of a given point, using a dataset. This is an implementation of using numpy.

**Usage:**

# Choice of bandwidth
bandwidth = 2000

Estimator = KDE(kernel="univariate_gaussian",univariate_bandwidth=bandwidth)
Estimator.fit(x_train)

# The bigger batch_size, the faster the operation goes but more ram-dependent
# If run out ram, please kindly decrease batch_size
est_density = Estimator.predict(x_test,y_test,batch_size=100)

```
