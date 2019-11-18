# Variational Inference for Mixtures of Gamma Distributions

This package implements variational inference for mixtures of gamma distributions. For more information, see *Comparing Markov Chain Monte Carlo and Variational Methods for Bayesian Inference on Mixtures of Gamma Distributions*.

Two parameterisations of the gamma distribution are available: the shape-rate parameterisation and the shape-mean parameterisation. The shape-mean parameterisation is recommended since it generally produces superior posterior approximations and predictive distributions.

## Getting Started

### Prerequisites

```
python>=3.6
tensorflow>=2.0
tensorflow_probability>=0.8
```

### Installing

I recommend installing using pip as it will also install the prerequisites. Run

```
pip install mix_gamma_vi
```

### Example

This is an example of posterior inference on a mixture of two gamma distributions under the shape-mean parameterisation. Suppose we have a one-dimensional tensor `x` of data.

```python
# import mix_gamma_vi function
from mix_gamma_vi import mix_gamma_vi

# Fit a model
fit = mix_gamma_vi(x, K=2)

# Get the fitted distribution
distribution = fit.distribution()

# Print the means of the parameters under the fitted distribution
distribution.mean()
``` 
```
{'pi': <tf.Tensor: id=4201, shape=(1, 2), dtype=float32, numpy=array([[0.50948393, 0.49051604]], dtype=float32)>,
 'beta': <tf.Tensor: id=4208, shape=(1, 2), dtype=float32, numpy=array([[1.0013412, 1.9965338]], dtype=float32)>,
 'alpha': <tf.Tensor: id=4212, shape=(1, 2), dtype=float32, numpy=array([[20.712543, 82.77388 ]], dtype=float32)>}
```

We can sample from this distribution by calling `distribution.sample()`.

For a more in-depth example, see [`example.py`](https://github.com/IsaacBreen/MixGammaVI/blob/master/example.ipynb).

## Performance Tip

To avoid retracing the tensor graph every time you change the parameters, pass them as TensorFlow constants. e.g. instead of the above, do

```python
fit = mix_gamma_vi(x, K=tf.constant(2))
```

## Authors

This work is submitted by Isaac Breen in partial fulfillment of the requirements for the Bachelor of Science degree with Honours at the University of Western Australia. Supervised by John Lau and Edward Cripps.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
