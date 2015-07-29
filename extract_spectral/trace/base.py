from astropy.modeling import Model, Parameter
from astropy.modeling.models import Polynomial1D, Gaussian1D


class GaussianPolynomialTrace(Polynomial1D):
    inputs = ('x', 'y')
    outputs = ('grid')

    sigma = Parameter()

    def __init__(self, degree, **params):
        super(GaussianPolynomialTrace, self).__init__(degree, domain=[-1, 1], window=[-1, 1], n_models=None,
                 model_set_axis=None, name=None, meta=None, **params)


    def evaluate(self, x, y, sigma, *coefs):
        return Gaussian1D.evaluate(y, 1., super(GaussianPolynomialTrace, self).evaluate(x, *coefs), sigma)