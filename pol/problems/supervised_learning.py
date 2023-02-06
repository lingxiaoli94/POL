import torch
import numpy as np
from .problem_base import ProblemBase
from torch.utils.data import DataLoader, TensorDataset


'''
This class represents the family of problems where the
evaluation is stochastic and can only be done using minibatches.
'''
class SupervisedLearningProblem(ProblemBase):
    def __init__(self,
                 device,
                 weight_dim,
                 loss_fn,
                 dataset,
                 eval_batch_size,
                 train_thetas,
                 weight_bbox):
        '''
        weight_dim: dimension of weights of the model (=D)
        loss_fn: (X, Y, W, thetas) -> L, where:
            X, Y are batches of inputs (...xX) and labels (...xY),
            W is a batch of weights (...xD),
            thetas are latent codes (...xC), and L is the resulting loss (...).
        dataset: an instance of torch.utils.data.Dataset
        train_thetas: NxC
        weight_bbox: Dx2, bbox of prior weights.
        '''
        self.device = device
        self.weight_dim = weight_dim
        self.loss_fn = loss_fn
        self.dataset = dataset
        self.data_loader = DataLoader(dataset,
                                      batch_size=eval_batch_size,
                                      pin_memory=True,
                                      shuffle=True,
                                      drop_last=False)
        self.data_itr = iter(self.data_loader)
        self.train_thetas = train_thetas
        self.weight_bbox = weight_bbox.to(device)

    def get_train_dataset(self):
        return TensorDataset(self.train_thetas)

    def sample_prior(self, data_batch, batch_size):
        '''
        Return: IxBxD
        '''
        thetas = data_batch[0]
        W = torch.rand([thetas.shape[0], batch_size, self.weight_dim],
                       device=self.device)  # IxBxD
        return (W * (self.weight_bbox[:, 1] - self.weight_bbox[:, 0]) +
                self.weight_bbox[:, 0])

    def do_satisfy(self, data_batch, W):
        '''
        W: IxBxD
        Return: IxB
        '''
        mask = torch.logical_and(W >= self.weight_bbox[:, 0],
                                 W <= self.weight_bbox[:, 1]) # IxBxD
        return mask.all(-1)

    def eval_loss(self, data_batch, W):
        '''
        W: IxB1xD
        Return: IxB1
        '''
        thetas = data_batch[0]  # IxC
        thetas = thetas.unsqueeze(1).repeat(1, W.shape[1], 1)  # IxB1xC

        try:
            X, Y = next(self.data_itr)  # B2xX, B2xY
        except StopIteration:
            self.data_itr = iter(self.data_loader)
            X, Y = next(self.data_itr)

        X = X.to(self.device)
        Y = Y.to(self.device)

        loss = self.eval_loss_batch(X, Y, W, thetas)

        return loss.mean(-1)

    def eval_loss_batch(self, X, Y, W, thetas):
        '''
        X: B2xX, Y: B2xY, W: IxB1xD, thetas: IxB1xC
        Return: IxB1xB2
        '''
        X = X.unsqueeze(0).unsqueeze(0).expand(
            thetas.shape[0], thetas.shape[1], -1, -1)  # IxB1xB2xX
        Y = Y.unsqueeze(0).unsqueeze(0).expand(
            thetas.shape[0], thetas.shape[1], -1, -1)  # IxB1xB2xY
        thetas = thetas.unsqueeze(2).expand(-1, -1, X.shape[2], -1)  # IxB1xB2xC
        W = W.unsqueeze(2).expand(-1, -1, X.shape[2], -1)  # IxB1xB2xD

        loss = self.loss_fn(X, Y, W, thetas)  # IxB1xB2
        return loss

    def eval_loss_whole(self, W, thetas):
        '''
        Non-stochastic version of eval_loss, used in evaluation.
        W: IxB1xD
        thetas: IxC
        Return: IxB1
        '''
        thetas = thetas.unsqueeze(1).repeat(1, W.shape[1], 1)  # IxB1xC
        total_loss = 0
        count = 0
        for X, Y in iter(self.data_loader):
            X_d = X.to(self.device)
            Y_d = Y.to(self.device)
            loss = self.eval_loss_batch(X_d, Y_d, W, thetas)  # IxB1xB2
            total_loss += loss.sum(-1).detach()  # IxB1
            count += loss.shape[-1]
        return total_loss / count

    def extract_latent_code(self, data_batch):
        '''
        Return: IxC
        '''
        return data_batch[0]

    def validate(self, solver, aux_data):
        '''
        aux_data should contain:
            * test_thetas
            * (optional) include_loss
            * (optional) include_witness
            * (optional) withness_batch_size
            * (optional) loss_landscape, a dictionary containing:
                - points, WxHx2, on which we evaluate the loss
        '''
        from pol.utils.validation.shortcuts import simple_validate

        def info_fn(thetas):
            result = {
                'type': 'ndarray_dict',
                'thetas': thetas.detach().cpu()
            }
            if aux_data.get('include_witness', False):
                for i in range(10):
                    witness = self.sample_prior([thetas],
                                                aux_data.get('witness_batch_size', 1024))
                    result[f'witness_{i}'] = witness.detach().cpu()

            landscape = aux_data.get('loss_landscape', None)
            if landscape is not None:
                from pol.utils.validation.shortcuts import eval_landscape
                result.update(eval_landscape(thetas=thetas,
                                             landscape=landscape,
                                             eval_fn=lambda D, X: self.eval_loss_whole(X, D[0])))
            return result

        def result_fn(thetas, X):
            result = {
                'type': 'ndarray_dict',
                'W': X.detach().cpu()
            }
            if aux_data.get('include_loss', True):
                result['loss'] = self.eval_loss_whole(X, thetas).detach().cpu()
                result['satisfy'] = self.do_satisfy([thetas], X).detach().cpu()
            return result

        return simple_validate(self, solver, aux_data, info_fn, result_fn)
