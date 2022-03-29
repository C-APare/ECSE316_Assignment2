import argparse
import math
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from dft_def import Definitions


def main():
    try:
        args = parse_args()
    except BaseException:
        print("ERROR\tIncorrect input syntax: Check arguments")
        return

    mode = args.mode
    image = args.image
    im = get_resized_image(image)

    if mode == 1:
        first_mode(im)
    elif mode == 2:
        second_mode(im)
    elif mode == 3:
        third_mode(im)
    elif mode == 4:
        fourth_mode()
    else:
        print("ERROR\tMode is not valid. Try again")

def get_new_shape(n):
    power = int(math.log(n, 2))
    return int(pow(2, power+1))

def first_mode(image: np.ndarray) -> None:
    #Get and resize image
    actual_image = plt.imread(image).astype(float)
    actual_shape = actual_image.shape
    new_shape  = get_new_shape(actual_shape[0]), get_new_shape(actual_shape[1])
    new_image = np.zeros(new_shape)
    new_image[:actual_shape[0], :actual_shape[1]] = actual_image

    #Do 2d fft
    fft_2d = Definitions.fft_dft_2d(new_image)

    #Display plots
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(new_image[:actual_shape[0], :actual_shape[1]], plt.cm.gray)
    ax[0].set_title('original')
    ax[1].imshow(np.abs(fft_2d), norm=colors.LogNorm())
    ax[1].set_title('fft 2d with lognorm')
    fig.suptitle('Mode 1')
    plt.show()

def second_mode(image: np.ndarray) -> None:
    #Get and resize image
    i=1

def third_mode(image: np.ndarray) -> None:
    i=1

def fourth_mode() -> None:
    i=1

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', action='store', dest='mode',
                        help='Mode of operation 1-> fast, 2-> denoise, 3-> compressing 4-> plot', type=int, default=1)
    parser.add_argument('-i', action='store', dest='image',
                        help='filepath of image to take DFT', type=str, default='moonlanding.png')
    return parser.parse_args()

if __name__ == "__main__":
    main()