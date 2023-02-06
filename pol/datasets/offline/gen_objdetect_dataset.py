import numpy as np
from pathlib import Path
from PIL import Image
from pol.utils.validation.scene_saver import load_scenes, count_h5_keys
import fiftyone as fo
from tqdm import tqdm
import argparse
from sklearn.cluster import MeanShift
from pol.utils.path import PathHelper
from pol.utils.cluster_mso import cluster_solutions
import json
import re

g_default_label = 'thing'
g_clustering_bandwidth = 0.02
g_use_density_as_confidence = False
g_max_cluster_count = 10

def create_dataset_from_scenes(dataset_name,
                               scene_path, img_dir_path,
                               prefix='scene_{:05}.png',
                               overwrite=False,
                               scene_range=None):
    if (dataset_name in fo.list_datasets()) and not overwrite:
        return fo.load_dataset(dataset_name)

    scenes = load_scenes(scene_path, scene_ids=scene_range)
    img_dir_path = Path(img_dir_path)
    img_dir_path.mkdir(parents=True, exist_ok=True)

    samples = []
    for i, scene in tqdm(enumerate(scenes)):
        img = scene['gt']['img']
        img = np.transpose(img, (1, 2, 0))
        img_path = img_dir_path / Path(prefix.format(i))
        pil_img = Image.fromarray(img.astype(np.uint8))
        pil_img.save(img_path)

        sample = fo.Sample(filepath=img_path)
        gt_detections = []

        gt_boxes = scenes[i]['gt']['boxes']
        gt_box_masks = scenes[i]['gt']['box_masks']

        gt_boxes = gt_boxes[gt_box_masks, :]
        if gt_boxes.shape[0] == 0:
            continue

        for box in gt_boxes:
            x, y, w, h = box
            # Convert to coco format in fiftyone.
            x -= 0.5 * w
            y -= 0.5 * h
            gt_detections.append(
                fo.Detection(bounding_box=[x, y, w, h], label=g_default_label)
            )
        sample['ground_truth'] = fo.Detections(detections=gt_detections)

        num_itr = count_h5_keys(scenes[i], 'itr')

        for itr in [num_itr-1]:
            if itr >= num_itr:
                continue
            pred_boxes = scenes[i][f'itr_{itr}']['boxes']
            pred_boxes, pred_freq = cluster_solutions(pred_boxes,
                                                      bandwidth=g_clustering_bandwidth, include_freq=True)
            pred_detections = []
            if pred_boxes is not None:
                for j, box in enumerate(pred_boxes):
                    x, y, w, h = box
                    # Convert to coco format in fiftyone.
                    x -= 0.5 * w
                    y -= 0.5 * h
                    pred_detections.append(
                        fo.Detection(bounding_box=[x, y, w, h], label=g_default_label,
                                    confidence=pred_freq[j])
                    )
            sample[f'pred_{itr}'] = fo.Detections(detections=pred_detections)

        samples.append(sample)

    dataset = fo.Dataset(dataset_name, overwrite=True)
    dataset.add_samples(samples)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--scene_range', type=int, nargs='+', default=None)
    parser.add_argument('--report', action='store_true', default=False)
    parser.add_argument('--serve', action='store_true', default=False)
    parser.add_argument('--port', type=int, default=5151)
    args = parser.parse_args()

    problem_name = args.problem
    method_name = args.method
    dataset_name = f'val_{problem_name}_{method_name}'
    path_helper = PathHelper('.')
    scene_path = path_helper.locate_scene_h5(problem_name,
                                             method_name)
    exp_name = path_helper.format_exp_name(problem_name, method_name)
    img_dir_path = path_helper.app_dir / 'images' / exp_name

    fo_dataset = create_dataset_from_scenes(dataset_name, scene_path, img_dir_path,
                                            overwrite=args.overwrite,
                                            scene_range=args.scene_range)
    fo_dataset.persistent = True

    if args.report:
        field_names = fo_dataset.get_field_schema().keys()
        itrs = []
        for name in field_names:
            m = re.match('pred_([0-9]+)', name)
            if m is not None:
                itrs.append(int(m.group(1)))
        itrs.sort()
        report_dict = {}
        for itr in itrs:
            if itr != itrs[-1] and itr <= 1: continue
            results = fo_dataset.evaluate_detections(
                f'pred_{itr}',
                gt_field='ground_truth',
                eval_key=f'eval_{itr}',
                compute_mAP=False,
            )
            metrics = results.metrics()
            report_dict[itr] = {
                'precision': metrics['precision'],
                'recall': metrics['recall']
            }
        json_obj = json.dumps(report_dict, indent=4)
        print('======Report for {}======'.format(exp_name))
        print(json_obj)
        json_path = img_dir_path / 'report.json'
        with open(json_path, 'w') as f:
            f.write(json_obj)

    if args.serve:
        session = fo.launch_app(fo_dataset, port=args.port)
        session.wait()
