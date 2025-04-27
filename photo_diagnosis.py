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
    print(f"Noise level: {noise_level:.2f}")
    return noise_level > 500

def detect_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Blur level (Laplacian Variance): {laplacian_var:.2f}")
    return laplacian_var < 100

def detect_low_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_pixel, max_pixel = np.min(gray), np.max(gray)
    print(f"Contrast range: {max_pixel - min_pixel}")
    return (max_pixel - min_pixel) < 100

def detect_color_imbalance(image):
    (b, g, r) = cv2.split(image)
    means = (np.mean(b), np.mean(g), np.mean(r))
    print(f"Color channel means: {means}")
    return np.std(means) > 20

# --- New Problems ---
def detect_overexposure(image):
    max_brightness = np.max(image)
    print(f"Max brightness: {max_brightness}")
    return max_brightness > 240

def detect_underexposure(image):
    min_brightness = np.min(image)
    print(f"Min brightness: {min_brightness}")
    return min_brightness < 30

def detect_red_eye(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_pixels = np.sum((hsv_image[:,:,0] < 10) & (hsv_image[:,:,1] > 100) & (hsv_image[:,:,2] > 50))
    print(f"Red-eye detection: {red_pixels}")
    return red_pixels > 10000

def detect_vignetting(image):
    height, width = image.shape[:2]
    corner_brightness = np.mean(image[:height//4, :width//4])  # top-left corner
    print(f"Vignetting corner brightness: {corner_brightness}")
    return corner_brightness < 50

def detect_over_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    max_saturation = np.max(hsv[:,:,1])
    print(f"Max saturation: {max_saturation}")
    return max_saturation > 200

# --- Fix Functions ---
def fix_overexposure(image):
    return cv2.convertScaleAbs(image, alpha=0.7, beta=-30)

def fix_underexposure(image):
    return cv2.convertScaleAbs(image, alpha=1.3, beta=50)

def fix_red_eye(image):
    b, g, r = cv2.split(image)
    r = cv2.subtract(r, 80)  # Decrease red channel value
    r[r < 0] = 0
    return cv2.merge([b, g, r])

def fix_vignetting(image):
    mask = np.ones_like(image)
    mask[:image.shape[0]//4, :image.shape[1]//4] = 1.5  # top-left corner
    return cv2.addWeighted(image, 0.8, mask, 0.2, 0)

def fix_over_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = hsv[:,:,1] * 0.7  # reduce saturation by 30%
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# --- Helper Fix Functions for Existing Problems ---
def denoise_image(image):
    # Using bilateral filter for better noise reduction with edge preservation
    #return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)

# OR try the alternative gentler NLM (if you want to try both)

# def denoise_image(image):
#     return

def sharpen_image(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def correct_color_balance(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- Main Processing Function ---
def process_image(image, filename):
    enhanced = image.copy()
    problems = []

    if detect_noise(image):
        enhanced = denoise_image(enhanced)
        problems.append('Noise detected and reduced')

    if detect_blur(image):
        enhanced = sharpen_image(enhanced)
        problems.append('Blur detected and sharpened')

    if detect_low_contrast(image):
        enhanced = enhance_contrast(enhanced)
        problems.append('Low contrast enhanced')

    if detect_color_imbalance(image):
        enhanced = correct_color_balance(enhanced)
        problems.append('Color imbalance corrected')

    if detect_overexposure(image):
        enhanced = fix_overexposure(enhanced)
        problems.append('Overexposure detected and fixed')

    if detect_underexposure(image):
        enhanced = fix_underexposure(enhanced)
        problems.append('Underexposure detected and fixed')

    if detect_red_eye(image):
        enhanced = fix_red_eye(enhanced)
        problems.append('Red-eye detected and fixed')

    if detect_vignetting(image):
        enhanced = fix_vignetting(enhanced)
        problems.append('Vignetting detected and fixed')

    if detect_over_saturation(image):
        enhanced = fix_over_saturation(enhanced)
        problems.append('Over-saturation detected and fixed')

    out_filename = f"output/enhanced_{filename}"
    cv2.imwrite(out_filename, enhanced)

    show_side_by_side(image, enhanced, filename, problems)

def show_side_by_side(original, enhanced, filename, problems):
    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    axs[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original')
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Enhanced')
    axs[1].axis('off')

    plt.suptitle(f"Problems fixed: {', '.join(problems)}")
    plt.savefig(f"output/compare_{filename}.png")
    plt.show()

def main():
    folder_path = "images"
    images, filenames = load_images_from_folder(folder_path)

    for img, fname in zip(images, filenames):
        print(f"\nProcessing {fname}...")
        process_image(img, fname)

if __name__ == "__main__":
    main()
