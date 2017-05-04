from sandbox.rocky.tf.algos.sensitive_lfd_npo import SensitiveLfD_NPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


class SensitiveLfD_TRPO(SensitiveLfD_NPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(SensitiveLfD_TRPO, self).__init__(optimizer=optimizer, **kwargs)
