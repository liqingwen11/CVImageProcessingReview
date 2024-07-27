import cv2
import numpy as np
import matplotlib.pyplot as plt

def homomorphic_filter(image_path, filter_size=31, sigma=10.0):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 将图像转换为浮点型并进行对数变换
    img_float = np.float32(img) + 1.0
    img_log = np.log(img_float)

    # 获取图像尺寸
    rows, cols = img_log.shape

    # 创建一个理想高通滤波器
    d0 = 30  # 截止频率
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    d = np.sqrt((x - cols / 2)**2 + (y - rows / 2)**2)
    H = 1 - np.exp(-d**2 / (2 * (d0 / 2.0)**2))

    # 将图像转换到频域
    img_dft = np.fft.fft2(img_log)
    img_dft_shifted = np.fft.fftshift(img_dft)

    # 应用同态滤波器
    img_dft_filtered = img_dft_shifted * H

    # 转换回空间域
    img_dft_filtered_shifted = np.fft.ifftshift(img_dft_filtered)
    img_filtered = np.fft.ifft2(img_dft_filtered_shifted)
    img_filtered = np.exp(np.real(img_filtered)) - 1

    # 归一化到[0, 255]范围
    img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)

    return img, img_filtered

# 使用同态滤波器
original_img, filtered_img = homomorphic_filter('your_image_path.jpg')

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(original_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Filtered Image')
plt.imshow(filtered_img, cmap='gray')
plt.axis('off')

plt.show()
