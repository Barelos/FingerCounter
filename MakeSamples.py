import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import sys
from numpy.fft import fftshift, ifftshift, fft2, ifft2

### HELPERS ###
def mid(n, add=False):
	half = np.floor(n/2).astype(np.int)
	return half + 1 if add else half


def magnitude(dx,dy):
	return np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)


def grid(n, m):
	high_n = mid(n) if n % 2 == 0 else mid(n) + 1
	high_m = mid(m) if m % 2 == 0 else mid(m) + 1
	return np.meshgrid(np.arange(-mid(m), high_m),np.arange(-mid(n), high_n))


### MAIN FUNC ###

def fourier_der(im):
	"""
	computes the magnitude of image derivatives using fourier_der
	param: im - grayscale image of type float64
	return: grayscale image of type float64
	"""
	n,m = im.shape
	u, v = grid(n, m)
	F = fftshift(fft2(im))
	FX = np.multiply(F,u)
	FY = np.multiply(F,v)
	dx = ifft2(ifftshift(FX)) * (2 * np.pi * 1j / m)
	dy = ifft2(ifftshift(FY)) * (2 * np.pi * 1j/ n)
	return magnitude(dx,dy)

def main():
    cap = cv2.VideoCapture(0)
    count = 1
    save = False
    folder = sys.argv[1]
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # to gray
        lut = np.zeros(256)
        lut[96:] = 1
        gray = lut[gray]
        gray = np.flip(gray, axis = 1)
        gray = fourier_der(gray)
        if save:
            n,m = gray.shape
            smaller = cv2.resize(gray, (0,0), fx=0.1, fy=0.1)
            plt.imsave(folder + str(count) + ".jpg", smaller, cmap='gray')
            count+= 1
        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('s'):
            save = False if save else True
            print("save mode: " + str(save))
        time.sleep(0.5)


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

main()
