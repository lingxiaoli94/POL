import torch
from pol.utils.batch_eval import batch_eval, batch_eval_index

def simple_validate(problem, solver, aux_data,
                    info_fn,
                    traj_fn):
    '''
    Here we assume the latent_code (a.k.a. theta) is simply the first entry of
    data_batch.
    aux_data should contain:
        * test_thetas, a pytorch tensor NxC
        * (universal only, optional) save_itrs = [100]
        * (universal only, optional) use_batch_eval = False
        * (universal only, optional) instance_batch_size = 32
        * (optional) prior_batch_size = 1024
        * (optional) fixed_prior_samples

    The output is a single scene corresponding to test_thetas.

    info_fn: thetas -> result, where result is an entry to be saved.
    info_fn decides what to save for the problem instances (e.g. thetas).

    traj_fn: (thetas, X) -> result, where:
        - thetas: IxC, X: IxBxD, will be on problem's device.
        - result: an entry to be saved in the scene dict.
    traj_fn decides what to save for each X on the trajectory.
    '''
    thetas = aux_data['test_thetas'].to(problem.device)
    scene = {}
    # Note: it is the responsibility of info_fn and traj_fn to
    # handle the case when the input's first dimension is too large to
    # fit in GPU.
    scene['info'] = info_fn(thetas)

    if solver.is_universal():
        fixed_prior_samples = aux_data.get('fixed_prior_samples', None)
        if fixed_prior_samples is not None:
            prior_samples = (
                fixed_prior_samples.to(problem.device).unsqueeze(0).expand(
                    thetas.shape[0], -1, -1))
        else:
            prior_samples = problem.sample_prior(
                [thetas], aux_data.get('prior_batch_size', 1024))

        X = prior_samples
        X_dict = {}
        X_dict[0] = X.cpu().detach()
        use_batch_eval = aux_data.get('use_batch_eval', False)
        save_itrs = aux_data.get('save_itrs', [100])
        max_itr = torch.as_tensor(save_itrs).max()
        for i in range(max_itr+1):
            X_detach = X.detach()
            if use_batch_eval:
                batch_size = aux_data.get('instance_batch_size', 32)
                X_cpu = batch_eval_index(
                    lambda inds: solver.extract_pushed_samples([thetas[inds]], X_detach[inds]),
                    thetas.shape[0],
                    dim=0,
                    batch_size=batch_size,
                    no_tqdm=True)
                X = X_cpu.to(solver.device)
            else:
                X = solver.extract_pushed_samples([thetas], X_detach)
            if i in save_itrs:
                X_dict[i] = X.detach().cpu()
    else:
        '''
        Otherwise the solver has extract_solutions() which returns a list.
        '''
        Xs = solver.extract_solutions()
        X_dict = {i: X.cpu() for i, X in enumerate(Xs)}

    for k, X in X_dict.items():
        result = traj_fn(thetas, X.to(problem.device))
        scene['itr_{}'.format(k)] = result

    return [scene]


def eval_landscape(thetas, landscape, eval_fn):
    '''
    Evaluate loss on a 2D landscape for visualization.
    eval_fn: same signature as ProblemBase.eval_loss
    '''
    assert(landscape is not None)
    P = landscape['points']  # WxHxD
    assert(P.shape[-1] == 2)
    P_flat = P.reshape(-1, 2).unsqueeze(0).expand(
        thetas.shape[0], -1, -1)  # IxWHxD
    P_flat = P_flat.to(thetas.device)
    P_loss = eval_fn([thetas], P_flat)  # IxWH
    P_loss = P_loss.reshape(thetas.shape[0], P.shape[0], P.shape[1])
    return {
        'landscape_P': P,
        'landscape_P_loss': P_loss.detach().cpu()
    }
