from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
import torch

class ProblemBase(ABC):
    @abstractmethod
    def get_train_dataset(self):
        '''
        Need to return a torch.utils.Dataset instance.
        '''
        pass

    def create_train_dataloader(self, batch_size):
        '''
        The problem can specify its own way of creating dataloader, e.g.,
        by supplying a custom collate_fn. If not a default dataloader will
        be created here.
        '''
        train_dataset = self.get_train_dataset()
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=min(len(train_dataset),
                                                     batch_size),
                                      pin_memory=True,
                                      shuffle=True,
                                      drop_last=True)
        return train_dataloader

    @abstractmethod
    def sample_prior(self, data_batch, batch_size):
        '''
        data_batch: a batch of the dataset as a list created from
        self.get_train_dataset. Each element should be a tensor with a same
        first dimension (=I) and on the same device as self.device.

        batch_size (=B): we need to sample this much prior points for each
        item in the batch.

        Return: a tensor of shape IxBxD, representing the IxB prior samples, each
        of dimension D (dimension of the variable we are optimization).
        '''
        pass

    @abstractmethod
    def eval_loss(self, data_batch, X):
        '''
        data_batch: current batch of data from either training or testing.
        We allow dataset to have different format. For instance, it could just
        be indices into the dataset.

        X: IxBxD, same format as the output of self.sample_prior
        Return: IxB
        '''
        pass

    @abstractmethod
    def extract_latent_code(self, data_batch):
        '''
        Each problem has its own way of extracting latent codes, potentially
        with a encoder specific to the problem.
        Return: IxC
        '''
        pass

    def validate(self, solver, aux_data=None):
        '''
        Return a list of scenes to be saved.

        Only the ProblemBase class knows what to save, so it needs to have access
        to the solver, which can be either universal or not.
        '''
        return []

    def apply_pre_push(self, data_batch, X):
        '''
        X: IxBxD
        Called before sending into the pushforward_net.
        '''
        return X

    def apply_projection(self, data_batch, X):
        '''
        X: IxBxD
        Sometimes the search space is constrained, so the problem can define its
        own projection operator to project points back to the constrained set.
        '''
        return X

    def push_forward(self, latent_code, X, pushforward_net):
        return pushforward_net(latent_code, X)

    def compute_distance(self, data_batch, X1, X2):
        '''
        X1, X2: IxBxD
        Return: IxB
        Problems can override the default L2 metric.
        '''
        return (X1 - X2).square().sum(-1)

    def do_satisfy(self, data_batch, X):
        '''
        X: IxBxD
        Return: IxB, a boolean tensor indicating if X satisfies the constraints.

        This is used only in evaluation, where we discard X that are out of range.
        '''
        return torch.ones(X.shape[:2], dtype=torch.bool, device=X.device)

    def get_nn_parameters(self):
        '''
        If the problem contains an encoder, then it should return its parameters
        here (to be put into the optimizer). Similarly for load/get_state_dict
        defined below.
        '''
        return None

    def load_state_dict(self, state_dict):
        return None

    def get_state_dict(self):
        return None

    def set_train_mode(self, is_training):
        pass

    def has_ground_truth(self):
        return False
