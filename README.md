# Fast-Fournier-Transformation
This is assignment 2 for Winter 2022 ECSE 316

## Documentation

### arguments
#### usage
python3 fft.py [-h] [-m MODE] [-i IMAGE]
Ex: python3 fft.py -m 3 -i moonlanding.png

#### optional arguments:
- -h, --help  show this help message and exit
- -m MODE     Mode of operation 1-> fast, 2-> denoise, 3-> compressing 4-> plot
- -i IMAGE    filepath of image to take FFT
  
### dft_def.py
- This class contains the dft definitions for one and two dimensions

### fft.py
- This is the main class

## Dependancies and external libraries used
- Python 3.8.8
- math
- argparse
- matplotlib
- numpy
- time
- statistics
