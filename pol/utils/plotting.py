import matplotlib.pyplot as plt
from pol.utils.validation.scene_saver import load_scenes, \
    count_h5_keys, find_max_h5_key
from pol.utils.path import find_newest_scene
from pathlib import Path
import torch
import numpy as np
import re

class LossTrajectoryPlotter:
    def __init__(self,
                 path_helper,
                 max_num_iter,
                 theta_name,
                 colors,
                 theta_range=None):
        self.path_helper = path_helper
        self.max_num_iter = max_num_iter
        self.theta_name = theta_name
        self.colors = colors
        self.theta_range = theta_range

        self.scene_cache = {}

    def load_exp_scene(self, problem, method):
        scene_path = self.path_helper.locate_scene_h5(problem, method)
        cached = self.scene_cache.get(scene_path, None)
        if cached is None:
            cached = load_scenes(scene_path, scene_ids=self.theta_range)[-1]
            self.scene_cache[scene_path] = cached
        return cached

    def plot_convergence(self, *,
                         theta_idx,
                         loss_name, satisfy_name,
                         problem, methods,
                         ax_traj, ax_hist,
                         num_hist_bin=50,
                         hist_max_rel=0,
                         all_satisfy=False,
                         loss_traj_lim=None,
                         method_itr={},
                         method_label_fn=None):
        def plot_traj(Ls, Ss, name, color):
            # Ls, Ss: array of B.
            num_t = len(Ls)
            traj_mean = []
            traj_std_lb = []
            traj_std_rb = []
            for i in range(num_t):
                L = torch.from_numpy(Ls[i])
                S = torch.from_numpy(Ss[i])
                if S.sum() == 0:
                    break
                else:
                    mean = L[S].mean()
                    std = (L[S] - mean).square().mean().sqrt()

                traj_mean.append(mean)
                traj_std_lb.append(mean-std)
                traj_std_rb.append(mean+std)
            ax_traj.plot(traj_mean, color=color,
                         label=name)
            ax_traj.fill_between(range(len(traj_std_lb)),
                                 traj_std_lb,
                                 traj_std_rb,
                                 alpha=0.5, color=color)
            if loss_traj_lim is not None:
                ax_traj.set_ylim(top=loss_traj_lim)

        loss_stats = []
        for method_idx, method in enumerate(methods):
            display_name = method if method_label_fn is None else method_label_fn(method)
            loss_itr = method_itr.get(method, -1)

            exp_name = self.path_helper.format_exp_name(problem, method)
            scene = self.load_exp_scene(problem, method)
            if method_idx == 0:
                theta = scene['info'][self.theta_name][theta_idx, :]

            Ls = []
            Ss = []
            max_iter = min(find_max_h5_key(scene, 'itr', return_itr=True),
                           self.max_num_iter+1)
            for k in range(max_iter+1):
                key = 'itr_{}'.format(k)
                if key not in scene:
                    continue
                tmp = scene[key]
                Ls.append(tmp[loss_name][theta_idx, :]) # B
                if all_satisfy:
                    Ss.append(np.ones_like(tmp[satisfy_name][theta_idx])) # B
                else:
                    Ss.append(tmp[satisfy_name][theta_idx]) # B
                if k == (loss_itr if loss_itr != -1 else max_iter):
                    loss_stats.append((Ls[-1], Ss[-1],
                                       display_name, self.colors[method]))

            if ax_traj is not None:
                plot_traj(Ls, Ss, display_name, self.colors[method])



        if ax_traj is not None:
            ax_traj.legend()

        if ax_hist is not None:
            # Next plot histograms.
            loss_lb = 1e100
            loss_rb = -1e100
            for L, S, _, _ in loss_stats:
                loss_lb = min(loss_lb, L[S].min())
                loss_rb = max(loss_rb, L[S].max())
            if hist_max_rel > 0:
                loss_rb = min(loss_rb, loss_lb * hist_max_rel)
            hist_bins = np.linspace(loss_lb, loss_rb, num_hist_bin)
            ax_hist.hist([L[S] for L, S, _, _ in loss_stats], hist_bins,
                         color=[s[3] for s in loss_stats],
                         label=[s[2] for s in loss_stats])
            ax_hist.set_xlabel(r'$f_\tau(x)$')
            ax_hist.set_ylabel('number of $x$')

            ax_hist.legend()
            ax_hist.set_title('$\\tau={}$'.format(','.join(
                ['{:.3f}'.format(t) for t in theta])))


    def plot_all_convergence(self, *,
                             theta_range,
                             fig_size=5,
                             include_hist=True,
                             include_traj=True,
                             **kwargs):
        if include_hist:
            fig_hist, ax_hist = plt.subplots(1, len(theta_range), squeeze=False)
        else:
            fig_hist, ax_hist = None, None
        if include_traj:
            fig_traj, ax_traj = plt.subplots(1, len(theta_range), squeeze=False)
        else:
            fig_traj, ax_traj = None, None

        for fig in [fig_hist, fig_traj]:
            if fig is not None:
                fig.set_figheight(fig_size / 2)
                fig.set_figwidth(fig_size*len(theta_range))
        for t, theta in enumerate(theta_range):
            self.plot_convergence(
                theta_idx=theta,
                ax_traj=ax_traj[0, t] if ax_traj is not None else None,
                ax_hist=ax_hist[0, t] if ax_hist is not None else None,
                **kwargs)


class WitnessMetricsPlotter:
    def __init__(self,
                 path_helper, *,
                 problem,
                 gt_method,
                 candidate_methods):
        self.path_helper = path_helper
        self.problem = problem
        self.gt_method = gt_method
        self.candidate_methods = candidate_methods

    def plot(self, colors, shortname_fn=None, threshold_ids=None):
        import json
        json_dicts = []
        for method in self.candidate_methods:
            eval_dir = self.path_helper.app_dir / 'eval' / self.path_helper.format_exp_name(self.problem, method)
            json_dict = json.loads(open(eval_dir / 'eval.json', 'r').read())
            json_dicts.append(json_dict)

        obj_thresholds = json_dicts[0]['obj_thresholds']
        if threshold_ids is not None:
            obj_thresholds = [obj_thresholds[i] for i in threshold_ids]
        precision_thresholds = json_dicts[0]['precision_thresholds']

        fig_size = 10
        fig, axes = plt.subplots(1, len(obj_thresholds), squeeze=False)
        fig.set_figheight(fig_size/2)
        fig.set_figwidth(fig_size*len(obj_thresholds))

        for i, obj_threshold in enumerate(obj_thresholds):
            for method, json_dict in zip(self.candidate_methods, json_dicts):
                result_dict = json_dict['result_list'][i]
                gt_count = result_dict['gt_count']['mean']
                candidate_count = result_dict['candidate_count']['mean']
                precision = result_dict['precision']
                precision_mean = torch.as_tensor(precision['mean'])
                precision_std = torch.as_tensor(precision['std'])
                divergence = result_dict['divergence']
                divergence_mean = divergence['mean']

                ax = axes[0, i]
                if shortname_fn is not None:
                    short_name = shortname_fn(method)
                else:
                    short_name = method
                label = short_name + r'($\rho=' + '{:.2f}'.format(candidate_count) + '$)'
                ax.plot(precision_thresholds, precision_mean,
                        label=label,
                       color=colors[method])
                ax.fill_between(precision_thresholds,
                                precision_mean - precision_std,
                                precision_mean + precision_std,
                                alpha=0.5, color=colors[method])
                ax.axvline(x=divergence_mean, color=colors[method], ls='--')
                ax.legend()

                ax.set_xlabel('$\delta$' + r'($\rho_{\textrm{gt}}=' + '{:.2f}'.format(gt_count) + ')$')
            y_label = r'$\textup{WP}_t^\delta$' + '($t={}$)'.format(obj_threshold)
            ax.set_ylabel(y_label)

        return fig
