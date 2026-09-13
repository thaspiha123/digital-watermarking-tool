import streamlit as st
import cv2
import numpy as np
import pywt
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------
# EMBED WATERMARK
# ---------------------------------------------------------
def embed_watermark(image, watermark):

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

    # Embed watermark into HL
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

    return watermarked_image


# ---------------------------------------------------------
# EXTRACT WATERMARK
# ---------------------------------------------------------
def extract_watermark(
    original_image,
    watermarked_image,
    watermark_shape
):

    # DWT of original image
    LL1, (LH1, HL1, HH1) = pywt.dwt2(
        original_image,
        "haar"
    )

    # DWT of watermarked image
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
# WATERMARK CORRELATION
# ---------------------------------------------------------
def calculate_correlation(original_watermark, extracted_watermark):

    # Convert both images to binary
    _, original_binary = cv2.threshold(
        original_watermark,
        127,
        1,
        cv2.THRESH_BINARY
    )

    _, extracted_binary = cv2.threshold(
        extracted_watermark,
        127,
        1,
        cv2.THRESH_BINARY
    )

    # Flatten images
    original_flat = original_binary.flatten().astype(float)
    extracted_flat = extracted_binary.flatten().astype(float)

    # Avoid invalid correlation
    if np.std(original_flat) == 0 or np.std(extracted_flat) == 0:
        return 0.0

    correlation = np.corrcoef(
        original_flat,
        extracted_flat
    )[0, 1]

    return float(correlation)


# ---------------------------------------------------------
# JPEG COMPRESSION ATTACK
# ---------------------------------------------------------
def jpeg_compression_attack(image, quality=50):

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )

    if not success:
        raise ValueError("JPEG compression failed.")

    attacked_image = cv2.imdecode(
        encoded,
        cv2.IMREAD_GRAYSCALE
    )

    return attacked_image


# ---------------------------------------------------------
# CROP ATTACK
# ---------------------------------------------------------
def crop_attack(image, crop_percent=20):

    height, width = image.shape

    crop_h = int(height * crop_percent / 100)
    crop_w = int(width * crop_percent / 100)

    cropped = image[
        crop_h:height - crop_h,
        crop_w:width - crop_w
    ]

    # Resize back to original dimensions
    cropped = cv2.resize(
        cropped,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    return cropped


# ---------------------------------------------------------
# GAUSSIAN NOISE ATTACK
# ---------------------------------------------------------
def gaussian_noise_attack(image, noise_strength=10):

    noise = np.random.normal(
        0,
        noise_strength,
        image.shape
    )

    noisy_image = image.astype(np.float32) + noise

    noisy_image = np.clip(
        noisy_image,
        0,
        255
    ).astype(np.uint8)

    return noisy_image


# ---------------------------------------------------------
# RESIZING ATTACK
# ---------------------------------------------------------
def resizing_attack(image, scale=0.5):

    height, width = image.shape

    new_width = int(width * scale)
    new_height = int(height * scale)

    # Reduce image size
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    # Resize back to original size
    resized = cv2.resize(
        resized,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    return resized


# ---------------------------------------------------------
# STREAMLIT PAGE
# ---------------------------------------------------------
st.set_page_config(
    page_title="Digital Watermarking Tool",
    page_icon="",
    layout="wide"
)

st.title("Digital Watermarking Tool")

st.write(
    "A DWT-based image watermarking tool for embedding, "
    "extracting and testing watermark robustness."
)


# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------
st.header("1. Upload Images")

original_file = st.file_uploader(
    "Upload Original Image",
    type=["jpg", "jpeg", "png"]
)

watermark_file = st.file_uploader(
    "Upload Watermark Image",
    type=["jpg", "jpeg", "png"]
)


# ---------------------------------------------------------
# LOAD IMAGES
# ---------------------------------------------------------
if original_file is not None and watermark_file is not None:

    original_bytes = np.asarray(
        bytearray(original_file.getvalue()),
        dtype=np.uint8
    )

    watermark_bytes = np.asarray(
        bytearray(watermark_file.getvalue()),
        dtype=np.uint8
    )

    original_image = cv2.imdecode(
        original_bytes,
        cv2.IMREAD_GRAYSCALE
    )

    original_watermark = cv2.imdecode(
        watermark_bytes,
        cv2.IMREAD_GRAYSCALE
    )

    if original_image is None:

        st.error("Could not load original image.")

    elif original_watermark is None:

        st.error("Could not load watermark image.")

    else:

        # -------------------------------------------------
        # DISPLAY INPUTS
        # -------------------------------------------------
        st.header("2. Input Images")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")

            st.image(
                original_image,
                use_container_width=True
            )

        with col2:
            st.subheader("Watermark")

            st.image(
                original_watermark,
                use_container_width=True
            )


        # -------------------------------------------------
        # EMBEDDING
        # -------------------------------------------------
        st.header("3. Embed Watermark")

        if st.button("Embed Watermark"):

            watermarked_image = embed_watermark(
                original_image,
                original_watermark
            )

            # Calculate PSNR
            psnr_value = cv2.PSNR(
                original_image,
                watermarked_image
            )

            # Calculate SSIM
            ssim_value = ssim(
                original_image,
                watermarked_image,
                data_range=255
            )

            # Store results
            st.session_state.watermarked_image = (
                watermarked_image
            )

            st.session_state.original_image = (
                original_image
            )

            st.session_state.original_watermark = (
                original_watermark
            )

            st.session_state.watermark_shape = (
                original_watermark.shape
            )

            st.session_state.psnr = psnr_value
            st.session_state.ssim = ssim_value

            st.success(
                "Watermark embedded successfully."
            )


        # -------------------------------------------------
        # DISPLAY WATERMARKED IMAGE
        # -------------------------------------------------
        if "watermarked_image" in st.session_state:

            st.header("4. Watermarked Image")

            st.image(
                st.session_state.watermarked_image,
                caption="Watermarked Image",
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "PSNR",
                    f"{st.session_state.psnr:.2f} dB"
                )

            with col2:

                st.metric(
                    "SSIM",
                    f"{st.session_state.ssim:.4f}"
                )


            # Download watermarked image
            success, encoded_image = cv2.imencode(
                ".png",
                st.session_state.watermarked_image
            )

            if success:

                st.download_button(
                    label="Download Watermarked Image",
                    data=encoded_image.tobytes(),
                    file_name="watermarked.png",
                    mime="image/png"
                )


            # -------------------------------------------------
            # EXTRACTION
            # -------------------------------------------------
            st.header("5. Extract Watermark")

            if st.button("Extract Watermark"):

                extracted_watermark = extract_watermark(
                    st.session_state.original_image,
                    st.session_state.watermarked_image,
                    st.session_state.watermark_shape
                )

                st.session_state.extracted_watermark = (
                    extracted_watermark
                )

                correlation = calculate_correlation(
                    st.session_state.original_watermark,
                    extracted_watermark
                )

                st.session_state.normal_correlation = (
                    correlation
                )


            if "extracted_watermark" in st.session_state:

                st.image(
                    st.session_state.extracted_watermark,
                    caption="Extracted Watermark",
                    use_container_width=False
                )

                st.metric(
                    "Watermark Correlation",
                    f"{st.session_state.normal_correlation:.4f}"
                )

                success, encoded_extracted = cv2.imencode(
                    ".png",
                    st.session_state.extracted_watermark
                )

                if success:

                    st.download_button(
                        label="Download Extracted Watermark",
                        data=encoded_extracted.tobytes(),
                        file_name="extracted_watermark.png",
                        mime="image/png"
                    )


            # -------------------------------------------------
            # ATTACK TESTING
            # -------------------------------------------------
            st.header("6. Attack Testing")

            st.write(
                "Test whether the watermark can still be "
                "recovered after the image is modified."
            )


            # -------------------------------------------------
            # JPEG COMPRESSION
            # -------------------------------------------------
            st.subheader("JPEG Compression Attack")

            jpeg_quality = st.slider(
                "JPEG Quality",
                min_value=10,
                max_value=90,
                value=50,
                step=10
            )

            if st.button("Apply JPEG Compression"):

                jpeg_image = jpeg_compression_attack(
                    st.session_state.watermarked_image,
                    jpeg_quality
                )

                st.session_state.jpeg_image = jpeg_image

                jpeg_extracted = extract_watermark(
                    st.session_state.original_image,
                    jpeg_image,
                    st.session_state.watermark_shape
                )

                st.session_state.jpeg_extracted = (
                    jpeg_extracted
                )

                jpeg_correlation = calculate_correlation(
                    st.session_state.original_watermark,
                    jpeg_extracted
                )

                st.session_state.jpeg_correlation = (
                    jpeg_correlation
                )


            if "jpeg_image" in st.session_state:

                st.image(
                    st.session_state.jpeg_image,
                    caption="JPEG Compressed Image",
                    use_container_width=True
                )

                st.image(
                    st.session_state.jpeg_extracted,
                    caption="Watermark Extracted After JPEG Attack",
                    use_container_width=False
                )

                st.metric(
                    "JPEG Watermark Correlation",
                    f"{st.session_state.jpeg_correlation:.4f}"
                )


            # -------------------------------------------------
            # CROPPING ATTACK
            # -------------------------------------------------
            st.subheader("Cropping Attack")

            crop_percent = st.slider(
                "Crop Percentage",
                min_value=5,
                max_value=40,
                value=20,
                step=5
            )

            if st.button("Apply Cropping Attack"):

                cropped_image = crop_attack(
                    st.session_state.watermarked_image,
                    crop_percent
                )

                st.session_state.cropped_image = (
                    cropped_image
                )

                cropped_extracted = extract_watermark(
                    st.session_state.original_image,
                    cropped_image,
                    st.session_state.watermark_shape
                )

                st.session_state.cropped_extracted = (
                    cropped_extracted
                )

                cropped_correlation = calculate_correlation(
                    st.session_state.original_watermark,
                    cropped_extracted
                )

                st.session_state.cropped_correlation = (
                    cropped_correlation
                )


            if "cropped_image" in st.session_state:

                st.image(
                    st.session_state.cropped_image,
                    caption="Cropped Image",
                    use_container_width=True
                )

                st.image(
                    st.session_state.cropped_extracted,
                    caption="Watermark Extracted After Cropping",
                    use_container_width=False
                )

                st.metric(
                    "Cropping Watermark Correlation",
                    f"{st.session_state.cropped_correlation:.4f}"
                )


            # -------------------------------------------------
            # GAUSSIAN NOISE ATTACK
            # -------------------------------------------------
            st.subheader("Gaussian Noise Attack")

            noise_strength = st.slider(
                "Noise Strength",
                min_value=5,
                max_value=40,
                value=10,
                step=5
            )

            if st.button("Apply Gaussian Noise Attack"):

                noisy_image = gaussian_noise_attack(
                    st.session_state.watermarked_image,
                    noise_strength
                )

                st.session_state.noisy_image = (
                    noisy_image
                )

                noise_extracted = extract_watermark(
                    st.session_state.original_image,
                    noisy_image,
                    st.session_state.watermark_shape
                )

                st.session_state.noise_extracted = (
                    noise_extracted
                )

                noise_correlation = calculate_correlation(
                    st.session_state.original_watermark,
                    noise_extracted
                )

                st.session_state.noise_correlation = (
                    noise_correlation
                )


            if "noisy_image" in st.session_state:

                st.image(
                    st.session_state.noisy_image,
                    caption="Image with Gaussian Noise",
                    use_container_width=True
                )

                st.image(
                    st.session_state.noise_extracted,
                    caption="Watermark Extracted After Gaussian Noise",
                    use_container_width=False
                )

                st.metric(
                    "Noise Watermark Correlation",
                    f"{st.session_state.noise_correlation:.4f}"
                )


            # -------------------------------------------------
            # RESIZING ATTACK
            # -------------------------------------------------
            st.subheader("Resizing Attack")

            resize_scale = st.slider(
                "Resize Scale",
                min_value=0.25,
                max_value=0.75,
                value=0.50,
                step=0.25
            )

            if st.button("Apply Resizing Attack"):

                resized_image = resizing_attack(
                    st.session_state.watermarked_image,
                    resize_scale
                )

                st.session_state.resized_image = (
                    resized_image
                )

                resize_extracted = extract_watermark(
                    st.session_state.original_image,
                    resized_image,
                    st.session_state.watermark_shape
                )

                st.session_state.resize_extracted = (
                    resize_extracted
                )

                resize_correlation = calculate_correlation(
                    st.session_state.original_watermark,
                    resize_extracted
                )

                st.session_state.resize_correlation = (
                    resize_correlation
                )


            if "resized_image" in st.session_state:

                st.image(
                    st.session_state.resized_image,
                    caption="Resized Image",
                    use_container_width=True
                )

                st.image(
                    st.session_state.resize_extracted,
                    caption="Watermark Extracted After Resizing",
                    use_container_width=False
                )

                st.metric(
                    "Resize Watermark Correlation",
                    f"{st.session_state.resize_correlation:.4f}"
                )


            # -------------------------------------------------
            # SUMMARY
            # -------------------------------------------------
            st.header("7. Performance Summary")

            st.write(
                "The following values show the quality of "
                "the watermarked image and watermark recovery."
            )

            summary_data = {
                "Metric": [
                    "PSNR",
                    "SSIM",
                    "Normal Extraction Correlation"
                ],
                "Value": [
                    f"{st.session_state.psnr:.2f} dB",
                    f"{st.session_state.ssim:.4f}",
                    (
                        f"{st.session_state.normal_correlation:.4f}"
                        if "normal_correlation"
                        in st.session_state
                        else "Not tested"
                    )
                ]
            }

            st.table(summary_data)

            st.write(
                "Higher PSNR and SSIM indicate better "
                "visual quality. A correlation value closer "
                "to 1 indicates better watermark recovery."
            )