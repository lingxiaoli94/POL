from pathlib import Path
import imageio
import argparse


class GIFGenerator:
    def __init__(self, img_folder, out_gif):
        self.img_folder = Path(img_folder)
        self.out_gif = out_gif

    def act(self, var_dict=None):
        files = sorted(self.img_folder.glob('*.png'))
        images = []
        for file in files:
            images.append(imageio.imread(file))
        imageio.mimsave(self.out_gif, images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_folder', type=str)
    parser.add_argument('out_gif', type=str)
    args = parser.parse_args()

    GIFGenerator(args.img_folder, args.out_gif).act()
