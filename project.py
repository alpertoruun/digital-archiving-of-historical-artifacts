import cv2
import numpy as np
import sqlite3
from datetime import datetime

def create_database(db_path):
    """ Veritabanı oluşturur ve gerekli tabloyu hazırlar. """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            file_path TEXT,
            process_date TEXT,
            additional_info TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_image_metadata(db_path, file_path, additional_info=""):
    """ Görüntü meta verilerini veritabanına ekler. """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO images (file_path, process_date, additional_info)
        VALUES (?, ?, ?)
    ''', (file_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), additional_info))
    conn.commit()
    conn.close()
def load_image(path):
    """ Renkli görüntüyü yükler. """
    return cv2.imread(path)

def reduce_noise(image, method='median', ksize=5):
    """ Görüntüdeki gürültüyü azaltır. """
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif method == 'median':
        return cv2.medianBlur(image, ksize)
    return image

def enhance_contrast(image):
    """ Görüntünün kontrastını artırır. """
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def adjust_color_balance(image, alpha=1.5, beta=0):
    """ Renk dengesini ayarlar. """
    return cv2.addWeighted(image, alpha, np.zeros_like(image, dtype=image.dtype), 0, beta)

def sobel_edge_detection(image, blur_ksize=5, sobel_ksize=5, skip_blur=False):
    """ Sobel kenar tespiti uygular. """
    if not skip_blur:
        image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    return sobel

def sharpen_image(image):
    """ Görüntüyü keskinleştirir. """
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)

def main():
    # Görüntüyü yükle
    image = load_image('/mnt/data/old_photo.jpg')

    # Gürültüyü azalt
    noise_reduced = reduce_noise(image)

    # Kontrastı artır
    contrast_enhanced = enhance_contrast(noise_reduced)

    # Renk dengesini ayarla
    color_adjusted = adjust_color_balance(contrast_enhanced)

    # Sobel kenar tespiti
    edges = sobel_edge_detection(color_adjusted)

    # Görüntüyü keskinleştir
    sharpened = sharpen_image(color_adjusted)  # Kenar tespiti sonucunu değil, renk ayarlı görüntüyü keskinleştiriyoruz

    # Sonuçları göster
    cv2.imshow('Original Image', image)
    cv2.imshow('Noise Reduced', noise_reduced)
    cv2.imshow('Contrast Enhanced', contrast_enhanced)
    cv2.imshow('Color Adjusted', color_adjusted)
    cv2.imshow('Sobel Edge Detection', edges)
    cv2.imshow('Sharpened Image', sharpened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Sonuçları kaydet
    processed_image_path = '/mnt/data/processed_image.jpg'
    cv2.imwrite(processed_image_path, sharpened)
    insert_image_metadata(db_path, processed_image_path, "Eski fotoğraf restorasyonu")
    cv2.imshow('Processed Image', sharpened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
