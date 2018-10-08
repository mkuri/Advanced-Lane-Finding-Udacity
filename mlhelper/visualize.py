from pathlib import Path
from typing import Iterable, Optional, Callable

import numpy as np
import matplotlib.pyplot as plt
import cv2

from matplotlib import rcParams
rcParams['font.family'] = 'Liberation Sans'
rcParams['font.size'] = 6

def combine_in_one_img(imgs: Iterable[np.ndarray],
                       titles: Iterable[str],
                       cmaps: Iterable[Optional[str]],
                       layout: str) -> plt.Figure:
    fig = plt.figure()
    
    for i, (img, title, cmap) in enumerate(zip(imgs, titles, cmaps)):
        ax = fig.add_subplot(int(layout+str(i+1)))
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)

    return fig


def combine_funced_imgs(img: np.ndarray,
                        funcs: Iterable[Callable[[np.ndarray], np.ndarray]],
                        titles: Iterable[str],
                        cmaps: Iterable[Optional[str]],
                        layout: str) -> plt.Figure:
    fig = plt.figure()

    for i, (func, title, cmap) in enumerate(zip(funcs, titles, cmaps)):
        ax = fig.add_subplot(int(layout+str(i+1)))
        ax.imshow(func(img), cmap=cmap)
        ax.set_title(title)

    return fig


def save_combined_imgs(img_paths: Iterable[Path],
                       funcs: Callable[[np.ndarray], np.ndarray],
                       titles: Iterable[str],
                       cmaps: Iterable[Optional[str]],
                       layout: str,
                       dst_path: str,
                       suffix: str,
                       ):
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig = combine_funced_imgs(img, funcs, titles, cmaps, layout)
        filename = dst_path + img_path.stem + suffix + img_path.suffix
        fig.savefig(filename, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def test():
    img1 = cv2.imread('../test_images/test1.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    img2 = cv2.imread('../output_images/binary/test1_binarized.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    imgs = [img1, img2]
    titles = ['Original', 'Binarized']
    cmaps = [None, 'gray']
    layout = '12'

    fig = combine_in_one_img(imgs, titles, cmaps, layout)
    plt.show()
    

if __name__ == '__main__':
    test()
