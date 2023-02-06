import torch


class UniformSampler:
    def __init__(self, device, low=[0, 0, 0], high=[1, 1, 1]):
        self.device = device
        self.low = torch.tensor(low).to(device)
        self.high = torch.tensor(high).to(device)

    def sample(self, batch_size):
        return self.low + ((self.high - self.low) *
                           torch.rand(batch_size, 3, device=self.device))
