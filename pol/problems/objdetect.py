from .problem_base import ProblemBase
import torch
import numpy as np
import random
from pol.datasets.objdetect import collate_padding_fn
from torch.utils.data import DataLoader
from pol.utils.device import transfer_data_batch
from pol.utils.box_distance import make_box_distance_fn
from pol.solvers.nn.resnet import ResNetEncoder

class ObjectDetection(ProblemBase):
    def __init__(self,
                 device,
                 train_dataset,
                 latent_dim,
                 use_regression,
                 box_loss_kind='l1',
                 box_loss_center_only=False,
                 sigmoid_projection=False,
                 encoder_kind='resnet',
                 encoder_params={}):
        r'''
        Args:
            train_dataset: map-style torch.utils.Dataset with items of the form
            {
                'img': CxHxW,
                'boxes': Kx4,
            }.
            The size of images and the number of boxes do not need to be the
            same for different items.
            Each box is represented as (center_x, center_y, width, height).
        '''
        self.device = device
        self.train_dataset = train_dataset
        self.latent_dim = latent_dim
        self.use_regression = use_regression
        self.box_loss_fn = make_box_distance_fn(box_loss_kind,
                                                box_loss_center_only)
        self.sigmoid_projection = sigmoid_projection
        if encoder_kind == 'resnet':
            self.encoder = ResNetEncoder(**encoder_params,
                                         hidden_dim=latent_dim)
        else:
            assert(False)
        self.encoder.to(device)
        from pol.utils.model_size import compute_model_size
        print('Encoder model size: {:3f}MB'.format(compute_model_size(self.encoder)))

    def set_train_mode(self, is_training):
        if is_training:
            self.encoder.train()
        else:
            self.encoder.eval()

    def get_train_dataset(self):
        return self.train_dataset

    def create_train_dataloader(self, batch_size):
        '''
            After collating, images and boxes will be padded to have same size,
            with an additional mask field for boxes:
            {
                'img': CxHxW,
                'boxes': Kx4,
                'box_masks': Kx4
            }.
        '''
        train_dataset = self.get_train_dataset()
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=min(len(train_dataset),
                                                     batch_size),
                                      collate_fn=collate_padding_fn,
                                      pin_memory=True,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=4)
        return train_dataloader

    def sample_prior(self, data_batch, batch_size):
        '''
        Returns: IxBx4

        We represent each bounding box internally as
            (center_x, center_y, width, height),
        same as in dataset, so everything is > 0.
        '''
        I = data_batch['img'].shape[0]
        return torch.cat([
            torch.rand([I, batch_size, 2], device=self.device),
            torch.rand([I, batch_size, 2], device=self.device),
        ], -1)


    def extract_latent_code(self, data_batch):
        return self.encoder(data_batch['img'])

    def has_ground_truth(self):
        return True

    def apply_projection(self, data_batch, X):
        if self.sigmoid_projection:
            return torch.sigmoid(X)
        else:
            return X

    def eval_loss(self, data_batch, X_pushed, X_prev, use_regression=None):
        '''
        Args:
            X_pushed, X_prev: IxBx4
        Returns:
            loss: IxB
        '''
        if use_regression is None:
            use_regression = self.use_regression
        boxes = data_batch['boxes'] # IxKx4
        masks = data_batch['box_masks'] # IxK

        if use_regression:
            X = X_prev
        else:
            X = X_pushed
        loss = self.box_loss_fn(X, boxes, share_dim=False) # IxBxK
        loss_filled = torch.where(masks.unsqueeze(-2).expand(-1, loss.shape[1], -1),
                                  loss,
                                  torch.full_like(loss, 1e8)) # IxBxK
        loss, closest_idx = loss_filled.min(-1) # IxB
        # Gather mask to handle the case when there is no gt bounding box.
        masks_gathered = torch.gather(masks, 1, closest_idx) # IxB
        if use_regression:
            closest_idx = closest_idx.unsqueeze(-1).expand(-1, -1, 4) # IxBx4
            boxes_gathered = torch.gather(boxes, 1, closest_idx) # IxBx4
            loss = self.box_loss_fn(X_pushed, boxes_gathered, share_dim=True) # IxB
        loss = torch.where(masks_gathered, loss, torch.zeros_like(loss)) # IxB

        return loss

    def get_nn_parameters(self):
        return self.encoder.parameters()

    def get_state_dict(self):
        return self.encoder.state_dict()

    def load_state_dict(self, d):
        self.encoder.load_state_dict(d)

    def validate(self, solver, aux_data):
        '''
        Args:
            solver: a solver instance
            aux_data: dict containing:
                * test_dataset, same format as train_dataset
                * instance_batch_size
                * prior_batch_size
                * num_itr
                * itr_whitelist
                * var_dict
                * (optional) val_seed
                * (optional) itr_whitelist
        '''
        assert(solver.is_universal())
        test_dataset = aux_data['test_dataset']
        val_seed = aux_data.get('val_seed', 567)
        num_itr = aux_data['num_itr']
        itr_whitelist = aux_data.get('itr_whitelist', range(num_itr + 1))

        def seed_worker(worker_id):
            np.random.seed(val_seed)
            random.seed(val_seed)

        test_dataloader = DataLoader(test_dataset,
                                     batch_size=aux_data['instance_batch_size'],
                                     collate_fn=collate_padding_fn,
                                     num_workers=1,
                                     worker_init_fn=seed_worker,
                                     shuffle=False)
        # One scene per test image..
        scenes = []
        total_size = 0
        total_loss = 0
        for t, data_batch in enumerate(test_dataloader):
            data_batch = transfer_data_batch(data_batch, self.device)
            latent_code = self.extract_latent_code(data_batch).detach()
            Xs = []

            from pol.solvers.fixed_number_solver import FixedNumberSolver
            from pol.solvers.frcnn import FRCNN
            if (isinstance(solver, FixedNumberSolver) or
                    isinstance(solver, FRCNN)):
                pred_boxes = solver.predict_boxes(data_batch)
                Xs.append(pred_boxes.cpu().detach())
                itr_whitelist = [0]
                prior_samples = None # HACK
            else:
                prior_samples = self.sample_prior(
                    data_batch, aux_data['prior_batch_size'])
                X = prior_samples
                Xs.append(X.cpu().detach())
                for j in range(num_itr):
                    X = solver.push_forward(data_batch, latent_code, X.detach())
                    Xs.append(X.cpu().detach())

            batch_size = data_batch['img'].shape[0]
            batch_scenes = [{} for i in range(batch_size)]
            for b in range(batch_size):
                img = data_batch['img_ori'][b, :, :, :]
                gt_boxes = data_batch['boxes'][b, :, :]
                box_masks = data_batch['box_masks'][b, :]
                batch_scenes[b]['gt'] = {
                    'type': 'ndarray_dict',
                    'img': img.detach().cpu(),
                    'boxes': gt_boxes.detach().cpu(),
                    'box_masks': box_masks.detach().cpu()
                }


            for j in itr_whitelist:
                if not (j < len(Xs)): continue
                boxes = Xs[j]
                for b in range(batch_size):
                    batch_scenes[b][f'itr_{j}'] = {
                        'type': 'ndarray_dict',
                        'boxes': boxes[b, :, :],
                    }
            scenes.extend(batch_scenes)

        return scenes


