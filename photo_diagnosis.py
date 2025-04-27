import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# Create output directory
if not os.path.exists('output'):
    os.makedirs('output')

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

# --- Problem Detection Functions ---
def detect_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise_level = np.var(gray)
    threshold = 150  # reduced threshold
    print(f"Noise level: {noise_level:.2f} (Threshold: {threshold})")
    return noise_level > threshold

def detect_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    threshold = 0.001 * gray.shape[0] * gray.shape[1]
    print(f"Blur (Laplacian Var): {lap_var:.2f} (Threshold: {threshold:.2f})")
    return lap_var < threshold

def detect_low_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.max(gray) - np.min(gray)
    print(f"Contrast: {contrast}")
    return contrast < 60

def detect_color_imbalance(image):
    (b, g, r) = cv2.split(image)
    means = (np.mean(b), np.mean(g), np.mean(r))
    std_dev = np.std(means)
    print(f"Color means: {means}, StdDev: {std_dev:.2f}")
    return std_dev > 15

def detect_overexposure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    percent_white = np.sum(gray > 240) / gray.size * 100
    print(f"Overexposed Pixels: {percent_white:.2f}%")
    return percent_white > 5

def detect_underexposure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    percent_black = np.sum(gray < 30) / gray.size * 100
    print(f"Underexposed Pixels: {percent_black:.2f}%")
    return percent_black > 5

def detect_red_eye(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_pixels = np.sum((hsv[:,:,0] < 10) & (hsv[:,:,1] > 100) & (hsv[:,:,2] > 50))
    print(f"Red-eye pixels: {red_pixels}")
    return red_pixels > 5000

def detect_vignetting(image):
    height, width = image.shape[:2]
    corners = [
        image[0:height//10, 0:width//10],
        image[0:height//10, -width//10:],
        image[-height//10:, 0:width//10],
        image[-height//10:, -width//10:]
    ]
    corner_brightness = np.mean([np.mean(c) for c in corners])
    center_brightness = np.mean(image[height//3:2*height//3, width//3:2*width//3])
    print(f"Corner brightness: {corner_brightness:.2f}, Center brightness: {center_brightness:.2f}")
    return (center_brightness - corner_brightness) > 30

def detect_over_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    high_saturation = np.sum(hsv[:,:,1] > 230) / hsv[:,:,1].size * 100
    print(f"Over-saturated pixels: {high_saturation:.2f}%")
    return high_saturation > 5

# --- Fix Functions ---
def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 1, 3, 7, 21)

def sharpen_image(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

def correct_color_balance(image):
    result = image.copy().astype(np.float32)
    b, g, r = cv2.split(result)
    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
    avg = (avg_b + avg_g + avg_r) / 3
    scale_b, scale_g, scale_r = avg / avg_b, avg / avg_g, avg / avg_r
    b = np.clip(b * scale_b, 0, 255)
    g = np.clip(g * scale_g, 0, 255)
    r = np.clip(r * scale_r, 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)

def fix_overexposure(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v * 0.8, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2BGR)

def fix_underexposure(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v * 1.3, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2BGR)

def fix_red_eye(image):
    b, g, r = cv2.split(image)
    mask = (r > 150) & (g < 80) & (b < 80)
    r[mask] = g[mask]
    return cv2.merge([b, g, r])

def fix_vignetting(image):
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.max(kernel)
    vignette = np.zeros_like(image)
    for i in range(3):
        vignette[:,:,i] = image[:,:,i] * (mask/255)
    return vignette

def fix_over_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 0.7, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2BGR)

# --- Main Processing Function ---
def process_image(image, filename):
    enhanced = image.copy()
    problems = []

    if detect_noise(image):
        enhanced = denoise_image(enhanced)
        problems.append('Noise reduced')

    if detect_blur(image):
        enhanced = sharpen_image(enhanced)
        problems.append('Sharpened')

    if detect_low_contrast(image):
        enhanced = enhance_contrast(enhanced)
        problems.append('Contrast enhanced')

    if detect_color_imbalance(image):
        enhanced = correct_color_balance(enhanced)
        problems.append('Color balance corrected')

    if detect_overexposure(image):
        enhanced = fix_overexposure(enhanced)
        problems.append('Overexposure fixed')

    if detect_underexposure(image):
        enhanced = fix_underexposure(enhanced)
        problems.append('Underexposure fixed')

    if detect_red_eye(image):
        enhanced = fix_red_eye(enhanced)
        problems.append('Red-eye corrected')

    if detect_vignetting(image):
        enhanced = fix_vignetting(enhanced)
        problems.append('Vignetting reduced')

    if detect_over_saturation(image):
        enhanced = fix_over_saturation(enhanced)
        problems.append('Saturation corrected')

    out_filename = f"output/enhanced_{filename}"
    cv2.imwrite(out_filename, enhanced)

    show_side_by_side(image, enhanced, filename, problems)

def show_side_by_side(original, enhanced, filename, problems):
    fig, axs = plt.subplots(1, 2, figsize=(14,7))
    axs[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original')
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Enhanced')
    axs[1].axis('off')

    plt.suptitle(f"Fixes: {', '.join(problems) if problems else 'No major problems detected.'}")
    plt.tight_layout()
    plt.savefig(f"output/compare_{filename}")
    plt.show()

def main():
    folder_path = "images"
    images, filenames = load_images_from_folder(folder_path)

    for img, fname in zip(images, filenames):
        print(f"\nProcessing {fname}...")
        process_image(img, fname)

if __name__ == "__main__":
    main()
