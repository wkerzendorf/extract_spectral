import numpy as np
from astropy.modeling import Model, Parameter
from astropy.modeling.models import Polynomial1D, Gaussian1D, Chebyshev1D
from scipy import optimize


class ModelFrame2D(Model):

    inputs = ()
    outputs = ('x', 'y')


    def __init__(self, observed_data):
        super(ModelFrame2D, self).__init__()
        self.grid_coordinates = np.ogrid[:observed_data.shape[0],
                                :observed_data.shape[1]]

    def evaluate(self, *args, **kwargs):
        return self.grid_coordinates


class GaussianPolynomialTrace(Model):
    """

    """
    inputs = ('x', 'y')
    outputs = ('frame', )

    sigma = Parameter()
    c0 = Parameter(default=0.)
    c1 = Parameter(default=0.)
    c2 = Parameter(default=0.)
    c3 = Parameter(default=1.)

    def __init__(self, degree, domain=[-1, 1], window=[-1, 1], **kwargs):
        super(GaussianPolynomialTrace, self).__init__(**kwargs)
        self.domain = domain
        self.window = window
        self.window_delta = self.window[1] - self.window[0]
        self.polynomial = np.polynomial.Chebyshev.basis(3, domain=domain)

    def evaluate(self, x, y, sigma, c0, c1, c2, c3):
        self.polynomial.coef = np.array([c0, c1, c2, c3])
        poly_eval = self.polynomial(x)
        poly_eval = (poly_eval + (self.window[0] + 1)) * self.window_delta / 2.
        return Gaussian1D.evaluate(y, 1., poly_eval, sigma)

class LinearLstSqExtraction(Model):
    inputs = ('frame', )
    outputs = ('frame', )

    def __init__(self, observed, observed_uncertainty=None):
        super(LinearLstSqExtraction, self).__init__()
        self.observed = observed
        if observed_uncertainty is None:
            self.observed_uncertainty = np.ones_like(self.observed)
        else:
            self.observed_uncertainty = observed_uncertainty
        self.amplitude = np.ones(self.observed.shape[0]) * np.nan
        self.sky = np.ones(self.observed.shape[0]) * np.nan

    def evaluate(self, frame):
        if frame.ndim == 3:
            frame = np.squeeze(frame)
        frame_output = np.empty_like(frame)
        sky_model = np.ones(frame.shape[1])

        for i in xrange(frame.shape[0]):
            synth_row = frame[i]
            observed_row = self.observed[i]
            A = np.vstack((synth_row, sky_model)).T
            amplitude, sky =  optimize.nnls(A, observed_row)[0]
            #amplitude = 0. if np.isinf(amplitude) else amplitude
            frame_output[i] = amplitude * frame[i] + sky
            self.amplitude[i] = amplitude
            self.sky[i] = amplitude
        return frame_output



class LogLikelihood(Model):
    inputs = ('frame', )
    outputs = ('loglikelihood', )


    def __init__(self, observed, observed_uncertainty=None):
        super(LogLikelihood, self).__init__()
        self.observed = observed
        if observed_uncertainty is None:
            self.observed_uncertainty = np.ones_like(self.observed)
        else:
            self.observed_uncertainty = observed_uncertainty

    def evaluate(self, frame):

        return -0.5 * np.sum(((self.observed - frame) /
                              self.observed_uncertainty)**2)

