import cv2
import numpy as np
import pywt
import os
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------
# EMBED WATERMARK
# ---------------------------------------------------------

def embed_watermark(image_path, watermark_path, output_path):

    # Load images in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Original image could not be loaded.")

    if watermark is None:
        raise ValueError("Watermark image could not be loaded.")

    # Apply DWT
    LL, (LH, HL, HH) = pywt.dwt2(image, "haar")

    # Resize watermark to HL size
    watermark = cv2.resize(
        watermark,
        (HL.shape[1], HL.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Convert watermark to binary
    _, watermark = cv2.threshold(
        watermark,
        127,
        1,
        cv2.THRESH_BINARY
    )

    # Watermark strength
    alpha = 10

    # Embed watermark into HL coefficients
    HL_watermarked = HL + alpha * watermark

    # Inverse DWT
    watermarked_image = pywt.idwt2(
        (LL, (LH, HL_watermarked, HH)),
        "haar"
    )

    # Convert to valid image range
    watermarked_image = np.clip(
        watermarked_image,
        0,
        255
    ).astype(np.uint8)

    # Ensure same dimensions
    watermarked_image = watermarked_image[
        :image.shape[0],
        :image.shape[1]
    ]

    # Save
    success = cv2.imwrite(
        output_path,
        watermarked_image
    )

    if not success:
        raise ValueError("Could not save the watermarked image.")

    return watermarked_image


# ---------------------------------------------------------
# EXTRACT WATERMARK
# ---------------------------------------------------------

def extract_watermark(
    original_image,
    watermarked_image,
    watermark_shape
):

    # Apply DWT to original
    LL1, (LH1, HL1, HH1) = pywt.dwt2(
        original_image,
        "haar"
    )

    # Apply DWT to watermarked image
    LL2, (LH2, HL2, HH2) = pywt.dwt2(
        watermarked_image,
        "haar"
    )

    # Difference between original and watermarked HL
    difference = HL2 - HL1

    # Watermark strength
    alpha = 10

    # Recover binary watermark
    extracted = np.where(
        difference > alpha / 2,
        255,
        0
    ).astype(np.uint8)

    # Resize extracted watermark
    extracted = cv2.resize(
        extracted,
        (watermark_shape[1], watermark_shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    return extracted


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    project_folder = os.path.dirname(
        os.path.abspath(__file__)
    )

    image_path = os.path.join(
        project_folder,
        "images",
        "original.jpg"
    )

    watermark_path = os.path.join(
        project_folder,
        "images",
        "watermark.png"
    )

    results_folder = os.path.join(
        project_folder,
        "results"
    )

    os.makedirs(
        results_folder,
        exist_ok=True
    )

    output_path = os.path.join(
        results_folder,
        "watermarked.png"
    )

    extracted_path = os.path.join(
        results_folder,
        "extracted_watermark.png"
    )

    # Load original
    original_image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    # Load watermark
    original_watermark = cv2.imread(
        watermark_path,
        cv2.IMREAD_GRAYSCALE
    )

    if original_image is None:
        raise ValueError(
            "Original image could not be loaded."
        )

    if original_watermark is None:
        raise ValueError(
            "Watermark image could not be loaded."
        )

    # Embed
    watermarked_image = embed_watermark(
        image_path,
        watermark_path,
        output_path
    )

    # PSNR
    psnr_value = cv2.PSNR(
        original_image,
        watermarked_image
    )

    # SSIM
    ssim_value = ssim(
        original_image,
        watermarked_image,
        data_range=255
    )

    # Extract
    extracted_watermark = extract_watermark(
        original_image,
        watermarked_image,
        original_watermark.shape
    )

    # Save extracted watermark
    cv2.imwrite(
        extracted_path,
        extracted_watermark
    )

    print()
    print("Watermark embedded successfully.")
    print("Watermarked image:", output_path)
    print("Extracted watermark:", extracted_path)
    print("PSNR:", psnr_value)
    print("SSIM:", ssim_value)