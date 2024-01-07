import cv2
import numpy as np
import sqlite3
import sys
from datetime import datetime
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print("Veritabanina bağlanildi.")
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn

def insert_image_metadata(conn, file_path, additional_info):
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO images (file_path, process_date, additional_info)
            VALUES (?, ?, ?)
        ''', (file_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), additional_info))
        conn.commit()
    except sqlite3.Error as e:
        print(e)


# Non-local Means Denoising için parametreleri ayarlayabilirsiniz.
def non_local_means_denoising(colored_image, h=10, hForColorComponents=10, templateWindowSize=7, searchWindowSize=21):
    return cv2.fastNlMeansDenoisingColored(colored_image, None, h, hForColorComponents, templateWindowSize, searchWindowSize)

# Renk dengesini ayarlamak için 'alpha' ve 'beta' parametrelerini ayarlayabilirsiniz.
def adjust_color_balance(image, alpha=1.5, beta=0):
    return cv2.addWeighted(image, alpha, np.zeros_like(image, dtype=image.dtype), 0, beta)

# Adaptif histogram eşitleme için 'clipLimit' ve 'tileGridSize' ayarlayabilirsiniz.
def adaptive_histogram_equalization(image, clipLimit=2.0, tileGridSize=(8, 8)):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# Unsharp mask için 'strength', 'radius' ve 'threshold' parametrelerini ayarlayabilirsiniz.
def unsharp_mask(image, strength=1.5, radius=5, threshold=3):
    blurred = cv2.GaussianBlur(image, (radius, radius), 0)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return sharpened

def load_image(path):
    return cv2.imread(path)

def main():
   # Görüntünün adresi ve database adresi
    database = "C:\\digital-archiving-of-historical-artifacts\\image_processing.db"
    image_path="C:\\digital-archiving-of-historical-artifacts\\photo2.jpg"
    conn = create_connection(database)
    image = load_image(image_path)
      # Renk dengesini ayarla
    color_adjusted = adjust_color_balance(image)

    # Gürültüyü azalt
    denoised = non_local_means_denoising(color_adjusted)

    # Kontrastı artır
    contrast_enhanced = adaptive_histogram_equalization(denoised)

    # Keskinleştir
    sharpened = unsharp_mask(contrast_enhanced)

    # Sonuçları göster
    cv2.imshow('Original Image', image)
    cv2.imshow('Color Adjusted', color_adjusted)
    cv2.imshow('Denoised Image', denoised)
    cv2.imshow('Contrast Enhanced', contrast_enhanced)
    cv2.imshow('Sharpened Image', sharpened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Sonuçları kaydet
    processed_image_path ="C:\\digital-archiving-of-historical-artifacts\photo2_processed.jpg"
    cv2.imwrite(processed_image_path, sharpened)
  
    if conn is not None:
        insert_image_metadata(conn, processed_image_path, "Eski fotograf restorasyonu")
        conn.close()

if __name__ == "__main__":
    main()