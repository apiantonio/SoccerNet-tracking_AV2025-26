import cv2
import numpy as np
import matplotlib.pyplot as plt

# path di un frame
# IMAGE_PATH = r"tracking\train\104\img1\000104.jpg" 
IMAGE_PATH = r"tracking\train\067\img1\000067.jpg" 

def generate_visual_steps(img_path):
    frame = cv2.imread(img_path)
    if frame is None:
        print("Errore: Immagine non trovata.")
        return

    # Replichiamo la logica interna di get_field_mask per estrarre gli step intermedi
    SCALE_FACTOR = 0.5
    small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    blurred = cv2.GaussianBlur(small_frame, (5, 5), 0)
    h_img, w_img = small_frame.shape[:2]

    # 1. HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    roi_hist = hsv[int(h_img * 0.4):, :, 0]
    hist = cv2.calcHist([roi_hist], [0], None, [180], [0, 180])
    hist[:30] = 0; hist[95:] = 0
    peak_hue = np.argmax(hist)
    if peak_hue == 0: peak_hue = 60
    lower = np.array([max(0, peak_hue - 18), 35, 30])
    upper = np.array([min(180, peak_hue + 18), 255, 255])
    mask_hsv = cv2.inRange(hsv, lower, upper)

    # 2. LAB
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    mask_lab = cv2.inRange(lab, np.array([0, 0, 0]), np.array([255, 118, 255]))
    
    # 3. Combined
    combined = cv2.bitwise_and(mask_hsv, mask_lab)

    # 4. Morph
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    morph = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_open, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, k_close, iterations=2)

    # 5. Convex Hull (Final)
    final_mask = np.zeros_like(morph)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        epsilon = 0.005 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        cv2.drawContours(final_mask, [approx], -1, 255, thickness=cv2.FILLED)

    # Plotting
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].imshow(frame_rgb)
    axes[0, 0].set_title("(a) Original Frame")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(combined, cmap='gray')
    axes[0, 1].set_title("(b) Combined Color Mask (HSV + LAB)")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(morph, cmap='gray')
    axes[1, 0].set_title("(c) After Morphology (Open/Close)")
    axes[1, 0].axis('off')

    # Overlay finale
    final_resized = cv2.resize(final_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = frame_rgb.copy()
    overlay[final_resized == 0] = [0, 0, 0] # Oscura ciò che è fuori campo
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("(d) Final Convex Hull Mask (Overlay)")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig("output/field_masking_process3.png", dpi=300)
    print("Immagine salvata in output/field_masking_process.png")
    plt.show()

if __name__ == "__main__":
    generate_visual_steps(IMAGE_PATH)