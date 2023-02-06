from pathlib import Path
import re

def find_newest_scene(parent_dir, regex='step-([0-9]+).h5'):
    parent_dir = Path(parent_dir)
    scenes = []
    for f in parent_dir.glob('*.h5'):
        m = re.search(regex, str(f))
        if m is not None:
            scenes.append(f)

    if len(scenes) == 0:
        # Non-universal.
        return parent_dir / 'result.h5'
    else:
        scenes.sort(key=lambda f: int(re.search(regex, str(f)).group(1)))
        return scenes[-1]

class PathHelper:
    def __init__(self, app_dir):
        self.app_dir = Path(app_dir)

    def format_exp_name(self, problem_name, method_name):
        return '{}/{}'.format(problem_name, method_name)

    def locate_scene_h5(self, problem_name, method_name):
        exp_name = self.format_exp_name(problem_name,
                                        method_name)
        parent_dir = self.app_dir / 'scenes' / exp_name
        if not (parent_dir.exists() and parent_dir.is_dir()):
            raise Exception('Cannot find {}!'.format(parent_dir))

        return find_newest_scene(parent_dir)
