import torch


class MH():
    """
    Implementation for the Metropolis Hasting (MH) algorithm
    using a gaussian kernel. Returns a list a samples from
    the target distribution.
    We work with the log form. !
    """
    def __init__(self, target: Callable[[torch.Tensor], torch.Tensor],
                 eq_steps: int,
                 num_samples: int,
                 dim: int,
                 step_size: float = 1.0
                 ):
        self.target = target
        self.eq_steps = eq_steps
        self.num_samples = num_samples
        self.dim = dim

    def generate_trial(self, state: torch.Tensor) -> torch.Tensor:
        sample = state + torch.randn_like(state)
        return sample

    def accept_decline(self, trial: torch.Tensor,
                       current_state: torch.Tensor) -> bool:
        alpha = self.target(trial) - self.target(current_state)
        if torch.rand(()) < torch.exp(torch.minimum(alpha, torch.tensor(0.0))):
            return True
        return False

    def sampler(self) -> torch.Tensor:
        # Thermalization
        x = torch.randn(self.dim)

        for _ in range(self.eq_steps):
            trial = self.generate_trial(x)
            if self.accept_decline(trial, x):
                x = trial

        # Sampling

        samples = torch.zeros(self.num_samples, self.dim)
        samples[0] = x

        for i in range(1, self.num_samples):
            trial = self.generate_trial(x)
            if self.accept_decline(trial, x):
                x = trial
            samples[i] = x

        return samples
