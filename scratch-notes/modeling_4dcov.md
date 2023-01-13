# Can we use ML to model 4D evolution of covariance?


In 4DEnVar, compute ensemble deviations `M(x_i^b) - M(\bar{x}^b)`
in order to estimate `X^b` and therefore `P^b`, the ensemble based background
covariance, for the next cycle.
In En4DVar this is done with the tangent linear model.
Can we use ML to bypass the model here, and obtain these deviations for the next
analysis cycle?

Some thoughts:
- Reservoir computing has been an absolute pain for capturing the entire
  nonlinear forward model, but would it have more success in capturing the
  perturbation evolution?
- Generative modeling seems like it could be a good approach too, since:
  1. GANs have been used successfully (?) for precipitation forecasting and
     downscaling (Ravuri et al, 2021; Harris et al, 2022, Duncan et al,
     preprint, 2022; Haupt et al, preprint, 2022)
  2. Diffusion models seem like GANs but easier to train (personal conversation
     with NREL scientist)
  3. Question is: can these models capture state dependence? This is the key
     property that we want.
