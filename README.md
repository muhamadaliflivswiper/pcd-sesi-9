# pcd-sesi-9

pip install imageio matplotlib numpy

import imageio
import numpy as np
import matplotlib.pyplot as plt

def roberts_operator(image):
    """Implementasi operator Roberts."""
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])

    grad_x = convolve2d(image, kernel_x)
    grad_y = convolve2d(image, kernel_y)

    return np.sqrt(grad_x*2 + grad_y*2)

def sobel_operator(image):
    """Implementasi operator Sobel."""
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad_x = convolve2d(image, kernel_x)
    grad_y = convolve2d(image, kernel_y)

    return np.sqrt(grad_x*2 + grad_y*2)

def convolve2d(image, kernel):
    """Operasi konvolusi 2D."""
    kernel_h, kernel_w = kernel.shape
    image_h, image_w = image.shape

    output = np.zeros_like(image)
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant')

    for i in range(image_h):
        for j in range(image_w):
            region = padded_image[i:i + kernel_h, j:j + kernel_w]
            output[i, j] = np.sum(region * kernel)

    return output

# Load gambar grayscale
image = imageio.imread('image.jpg', as_gray=True)

# Normalisasi gambar
image = image / 255.0

# Deteksi tepi menggunakan operator Roberts
edges_roberts = roberts_operator(image)

# Deteksi tepi menggunakan operator Sobel
edges_sobel = sobel_operator(image)

# Visualisasi hasil
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Gambar Asli")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Deteksi Tepi - Roberts")
plt.imshow(edges_roberts, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Deteksi Tepi - Sobel")
plt.imshow(edges_sobel, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
