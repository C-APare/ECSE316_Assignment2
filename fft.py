import argparse
import math
import statistics
import time
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from dft_def import Definitions


def main():
    try:
        args = parser()
    except Exception:
        print("ERROR\tIncorrect input syntax: Check arguments")
        return
    mode = args.mode
    image = args.image

    Definitions.test()

    if mode == 1:
        first_mode(image)
    elif mode == 2:
        second_mode(image)
    elif mode == 3:
        third_mode(image)
    elif mode == 4:
        fourth_mode()
    else:
        print("ERROR\tMode is not valid. Try again")

def get_new_shape(n):
    power = int(math.log(n, 2))
    return int(pow(2, power+1))

def first_mode(image: str) -> None:
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

def second_mode(image: str) -> None:
    # Define the ratio to keep for the frequencies
    keep_ratio = 0.9

    #Get and resize image
    actual_image = plt.imread(image).astype(float)
    actual_shape = actual_image.shape
    new_shape  = get_new_shape(actual_shape[0]), get_new_shape(actual_shape[1])
    new_image = np.zeros(new_shape)
    new_image[:actual_shape[0], :actual_shape[1]] = actual_image

    # Get fft 2D and remove the high frequency values
    fft_2d = Definitions.fft_dft_2d(new_image)
    row, column = fft_2d.shape
    fft_2d[int(row * keep_ratio) : int(row*(1-keep_ratio))] = 0
    fft_2d[:, int(column*keep_ratio) : int(column * (1-keep_ratio))] = 0

    # Inverse to denoise image
    fft_2d_inverse = Definitions.fft_dft_2d_inverse(fft_2d).real

    # Display plot
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(new_image[:actual_shape[0], :actual_shape[1]], plt.cm.gray)
    ax[0].set_title('original')
    ax[1].imshow(fft_2d_inverse[:actual_shape[0], :actual_shape[1]], plt.cm.gray)
    ax[1].set_title('denoised')
    fig.suptitle('Mode 2')
    plt.show()

def third_mode(image: str) -> None:
    # Define the compression levels
    compression_levels = [0, 10, 25, 40, 75, 95]

    #Get and resize image
    actual_image = plt.imread(image).astype(float)
    actual_shape = actual_image.shape
    new_shape  = get_new_shape(actual_shape[0]), get_new_shape(actual_shape[1])
    new_image = np.zeros(new_shape)
    new_image[:actual_shape[0], :actual_shape[1]] = actual_image
    first_count = actual_shape[0] * actual_shape[1]

    # Get fft
    fft = Definitions.fft_dft_1d(new_image)

    # Render
    fig, ax = plt.subplots(2, 3)
    for i in range(2):
        for j in range(3):
            compression_level = compression_levels[i*3 + j]
            compressed_image = compress_image(
                fft, compression_level, first_count)
            ax[i, j].imshow(np.real(compressed_image)[
                            :actual_shape[0], :actual_shape[1]], plt.cm.gray)
            ax[i, j].set_title('{}% compression'.format(compression_level))
    fig.suptitle('Mode 3')
    plt.show()

def compress_image(image: np.ndarray, compress: int, count: int) -> np.ndarray:
    rest = 100-compress
    upper_bound = np.percentile(image, 100 - rest//2)
    lower_bound = np.percentile(image, rest//2)

    # Print number of non-zeros
    print(f'non zero values for level {compress}% are {int(count * ((100 - compress) / 100.0))} out of {count}')

    compressed_image = image * np.logical_or(image <= lower_bound, image >= upper_bound)
    return Definitions.fft_dft_2d_inverse(compressed_image)

def fourth_mode() -> None:
    # define sample runs
        runs = 10

        # run plots
        fig, ax = plt.subplots()

        ax.set_xlabel('problem size')
        ax.set_ylabel('runtime in seconds')
        ax.set_title('Line plot with error bars')

        for algo_index, algo in enumerate([Definitions.naive_dft_2d, Definitions.fft_dft_2d]):
            print(f"starting measurement for {algo.__name__}")
            x = []
            y = []

            problem_size = 2**5
            while problem_size <= 2**10:
                print(f"doing problem size of {problem_size}")
                a = np.random.rand(int(math.sqrt(problem_size)),
                                   int(math.sqrt(problem_size)))
                x.append(problem_size)

                stats_data = []
                for i in range(runs):
                    print(f"run {i+1} ...")
                    start_time = time.time()
                    algo(a)
                    delta = time.time() - start_time
                    stats_data.append(delta)

                mean = statistics.mean(stats_data)
                sd = statistics.stdev(stats_data)

                print(f"for problem size of {problem_size} over {runs} runs: mean {mean}, stdev {sd}")

                y.append(mean)

                # ensure square and power of 2 problems sizes
                problem_size *= 4

            color = 'r--' if algo_index == 0 else 'g'
            plt.errorbar(x, y, yerr=sd, fmt=color)
        plt.show()

def parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='mode',
                        help='Mode of operation 1-> fast, 2-> denoise, 3-> compressing 4-> plot', type=int, default=1)
    parser.add_argument('-i', dest='image',
                        help='filepath of image to take FFT', default='moonlanding.png')
    return parser.parse_args()

if __name__ == "__main__":
    main()