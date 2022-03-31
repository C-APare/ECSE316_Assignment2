import numpy as np


class Definitions:
    
    @staticmethod
    def naive_dft_1d(image: np.ndarray) -> np.ndarray:
        im = np.asarray(image, dtype=complex)
        size = im.shape[0]
        out = np.zeros(size, dtype=complex)

        for i in range(size):
            for j in range(size):
                out[i] += im[j] * np.exp(-2j * np.pi * i * j / size)

        return out    

    @staticmethod
    def fft_dft_1d(image: np.ndarray) -> np.ndarray:
        im = np.asarray(image, dtype=complex)
        size = im.shape[0]
        print(size)
        if (size % 2 != 0):
            raise AssertionError("Error\tSize must be a power of 2")
        elif size <= 16:
            return Definitions.naive_dft_1d(image)
        else:
            even = Definitions.fft_dft_1d(image[::2])
            odd = Definitions.fft_dft_1d(image[1::2])
            out = np.zeros(size, dtype=complex)

            for n in range (size):
                out[n] = even[n % (size//2)] + np.exp(-2j * np.pi * n / size) * odd[n % (size//2)]
            
            
            return out

    @staticmethod
    def naive_dft_1d_inverse(image:np.ndarray) -> np.ndarray:
        im = np.asarray(image, dtype=complex)
        size = im.shape[0]
        out = np.zeros(size, dtype=complex)

        for i in range(size):
            for j in range(size):
                out[i] += im[j] * np.exp(2j * np.pi * i * j / size)

            out[i] /= size

        return out 

    @staticmethod
    def fft_dft_1d_inverse(image: np.ndarray) -> np.ndarray:
        im = np.asarray(image, dtype=complex)
        size = im.shape[0]
        if (size % 2 != 0):
            raise AssertionError("Error\tSize must be a power of 2")
        elif size <= 16:
            return Definitions.naive_dft_1d_inverse(image)
        else:
            even = Definitions.fft_dft_1d_inverse(image[::2])
            odd = Definitions.fft_dft_1d_inverse(image[1::2])
            out = np.zeros(size, dtype=complex)

            for n in range (size):
                out[n] = even[n % (size//2)] + np.exp(2j * np.pi * n / size) * odd[n % (size//2)]
                out[n] /= size

            return out

    @staticmethod
    def naive_dft_2d(image: np.ndarray) -> np.ndarray:
        im = np.asarray(image, dtype=complex)
        size1d, size2d = im.shape
        out = np.zeros((size1d, size2d), dtype=complex) 
        for i in range(size1d):
            for j in range(size2d):
                for m in range(size2d):
                    for n in range(size1d):
                        out[i,j] += im[n, m] * np.exp(-2j * np.pi * ((j * m / size2d) + (i * n /size1d)))
        return out

    @staticmethod
    def fft_dft_2d(image: np.ndarray) -> np.ndarray:
        im = np.asarray(image, dtype=complex)
        size1d, size2d = im.shape
        out = np.zeros((size1d, size2d), dtype=complex) 
        for col in range(size2d):
            out[:, col] = Definitions.fft_dft_1d(im[:, col])
        for row in range(size1d):
            out[row, :] = Definitions.fft_dft_1d(im[row, :])
        return out

    @staticmethod
    def fft_dft_2d_inverse(image: np.ndarray) -> np.ndarray:
        im = np.asarray(image, dtype=complex)
        size1d, size2d = im.shape
        out = np.zeros((size1d, size2d), dtype=complex) 

        for row in range(size1d):
            out[row, :] = Definitions.fft_dft_1d_inverse(im[row, :])
            

        for col in range(size2d):
            out[:, col] = Definitions.fft_dft_1d_inverse(im[:, col])

        return out

    @staticmethod
    def test():
        # one dimension
        a = np.random.random(1024)
        fft = np.fft.fft(a)

        # two dimensions
        a2 = np.random.rand(32, 32)
        fft2 = np.fft.fft2(a2)

        tests = (
            (Definitions.naive_dft_1d, a, fft),
            (Definitions.naive_dft_1d_inverse, fft, a),
            (Definitions.fft_dft_1d, a, fft),
            (Definitions.fft_dft_1d_inverse, fft, a),
            (Definitions.naive_dft_2d, a2, fft2),
            (Definitions.fft_dft_2d, a2, fft2),
            (Definitions.fft_dft_2d_inverse, fft2, a2)
        )

        for method, args, expected in tests:
            if not np.allclose(method(args), expected):
                print(args)
                print(method(args))
                print(expected)
                raise AssertionError(
                    "{} failed the test".format(method.__name__))