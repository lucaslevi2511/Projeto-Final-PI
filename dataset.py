import os
import glob
import cv2
import numpy as np

from config import TARGET_SIZE, PADDING_RATIO, MAKE_SQUARE
from image_processing import (
    white_balance_gray_world, rgb2ycbcr, ycbcr2rgb, 
    bilateral_filter, calculate_histogram, normalize_histogram,
    cumulative_distribution_function, histogram_equalization, normalizar
)
from compression import compress_numpy_image, decompress_to_numpy

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def parse_label_line(line):
    parts = line.strip().split()
    if len(parts) < 6:
        return None, None, None

    numeric = []
    i = 0
    while i < len(parts):
        try:
            numeric.append(float(parts[i]))
            i += 1
        except ValueError:
            break

    rest = parts[i:]
    class_name = rest[0] if len(rest) >= 1 else "unknown"
    class_id = rest[1] if len(rest) >= 2 else None

    if len(numeric) < 4 or len(numeric) % 2 != 0:
        return None, None, None

    pts = np.array(numeric, dtype=np.float32).reshape(-1, 2)
    return pts, class_name, class_id

def polygon_to_bbox(points, img_w, img_h, padding_ratio=0.10):
    x_min = int(np.floor(points[:, 0].min()))
    y_min = int(np.floor(points[:, 1].min()))
    x_max = int(np.ceil(points[:, 0].max()))
    y_max = int(np.ceil(points[:, 1].max()))

    bw = x_max - x_min
    bh = y_max - y_min
    pad = int(max(bw, bh) * padding_ratio)

    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(img_w, x_max + pad)
    y_max = min(img_h, y_max + pad)

    return x_min, y_min, x_max, y_max

def make_square_crop(img, x_min, y_min, x_max, y_max):
    h, w = img.shape[:2]
    bw = x_max - x_min
    bh = y_max - y_min
    side = max(bw, bh)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    x_min2 = cx - side // 2
    y_min2 = cy - side // 2
    x_max2 = x_min2 + side
    y_max2 = y_min2 + side

    if x_min2 < 0:
        x_max2 -= x_min2
        x_min2 = 0
    if y_min2 < 0:
        y_max2 -= y_min2
        y_min2 = 0
    if x_max2 > w:
        diff = x_max2 - w
        x_min2 = max(0, x_min2 - diff)
        x_max2 = w
    if y_max2 > h:
        diff = y_max2 - h
        y_min2 = max(0, y_min2 - diff)
        y_max2 = h

    return x_min2, y_min2, x_max2, y_max2

def process_image_and_labels(image_path, label_path, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Falha ao abrir imagem: {image_path}")
        return

    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        print(f"Label não encontrado para: {image_path}")
        return

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for idx, line in enumerate(lines):
        points, class_name, class_id = parse_label_line(line)
        if points is None:
            continue

        x_min, y_min, x_max, y_max = polygon_to_bbox(points, w, h, PADDING_RATIO)

        if MAKE_SQUARE:
            x_min, y_min, x_max, y_max = make_square_crop(img, x_min, y_min, x_max, y_max)

        crop = img[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue

        crop_resized = cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)

        class_dir = os.path.join(output_dir, class_name)
        ensure_dir(class_dir)

        out_name = f"{base_name}_{idx}.jpg"
        
        # Pipeline de processamento de imagem
        crop_resized = crop_resized[..., [2, 1, 0]] # BGR to RGB
        crop_resized = white_balance_gray_world(crop_resized)
        ycbcr_img = rgb2ycbcr(crop_resized)

        cb = ycbcr_img[:, :, 1]
        cr = ycbcr_img[:, :, 2]
        y = ycbcr_img[:, :, 0].astype(np.float32) / 255.0

        y_filtered = bilateral_filter(y, 2, 0.1)
        y_filtered = np.clip(y_filtered * 255, 0, 255).astype(np.uint8)

        hist = calculate_histogram(y_filtered)
        norm_hist = normalize_histogram(hist, y_filtered.size)
        cdf = cumulative_distribution_function(norm_hist)
        y_equalized = histogram_equalization(y_filtered, cdf)
        
        ycbcr_equalized = np.dstack((y_equalized, cb, cr))
        img_equalized = ycbcr2rgb(ycbcr_equalized)
        cnn_img_equalized = normalizar(img_equalized)
        img_to_save = (cnn_img_equalized * 255).astype(np.uint8)

        # Compressão e Descompressão (Simulação/Ciclo completo)
        shape = img_to_save.shape
        img_comprimida = compress_numpy_image(img_to_save)
        img_descomprimida = decompress_to_numpy(img_comprimida, shape)
        
        out_path = os.path.join(class_dir, out_name)
        cv2.imwrite(out_path, img_descomprimida)

def extrair_recortes(root_dir, output_dir, process_all=False, max_images=50):
    ensure_dir(output_dir)
    subsets = ["train", "valid", "test"]
    dataset_files = []

    for subset in subsets:
        images_dir = os.path.join(root_dir, subset, "images")
        labels_dir = os.path.join(root_dir, subset, "labelTxt")

        if not os.path.exists(images_dir):
            continue

        image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(images_dir, ext)))

        for image_path in image_paths:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(labels_dir, base_name + ".txt")

            if os.path.exists(label_path):
                dataset_files.append((subset, image_path, label_path))

    print(f"\n[Dataset] Total de imagens originais encontradas: {len(dataset_files)}")

    if not process_all:
        dataset_files = dataset_files[:max_images]
        print(f"[Dataset] Processando apenas um subset de {len(dataset_files)} imagens para teste rápido.")

    for subset, image_path, label_path in dataset_files:
        process_image_and_labels(image_path, label_path, os.path.join(output_dir, subset))
    
    print("[Dataset] Extração de recortes concluída com sucesso!")