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
        bytearray(original_file.read()),
        dtype=np.uint8
    )

    watermark_bytes = np.asarray(
        bytearray(watermark_file.read()),
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

            if "extracted_watermark" in st.session_state:

                st.image(
                    st.session_state.extracted_watermark,
                    caption="Extracted Watermark",
                    use_container_width=False
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

            if "jpeg_image" in st.session_state:

                st.image(
                    st.session_state.jpeg_image,
                    caption="JPEG Compressed Image",
                    use_container_width=True
                )

                if st.button(
                    "Extract Watermark After JPEG Attack"
                ):

                    jpeg_extracted = extract_watermark(
                        st.session_state.original_image,
                        st.session_state.jpeg_image,
                        st.session_state.watermark_shape
                    )

                    st.session_state.jpeg_extracted = (
                        jpeg_extracted
                    )

                if "jpeg_extracted" in st.session_state:

                    st.image(
                        st.session_state.jpeg_extracted,
                        caption="Watermark Extracted After JPEG Attack",
                        use_container_width=False
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

            if "cropped_image" in st.session_state:

                st.image(
                    st.session_state.cropped_image,
                    caption="Cropped Image",
                    use_container_width=True
                )

                if st.button(
                    "Extract Watermark After Cropping"
                ):

                    cropped_extracted = extract_watermark(
                        st.session_state.original_image,
                        st.session_state.cropped_image,
                        st.session_state.watermark_shape
                    )

                    st.session_state.cropped_extracted = (
                        cropped_extracted
                    )

                if "cropped_extracted" in st.session_state:

                    st.image(
                        st.session_state.cropped_extracted,
                        caption="Watermark Extracted After Cropping",
                        use_container_width=False
                    )

# GAUSSIAN NOISE ATTACK

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

    return noisy_imagegit add .git add .
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

    st.session_state.noisy_image = noisy_image

if "noisy_image" in st.session_state:

    st.image(
        st.session_state.noisy_image,
        caption="Image with Gaussian Noise",
        use_container_width=True
    )

    if st.button(
        "Extract Watermark After Noise Attack"
    ):

        noise_extracted = extract_watermark(
            st.session_state.original_image,
            st.session_state.noisy_image,
            st.session_state.watermark_shape
        )

        st.session_state.noise_extracted = (
            noise_extracted
        )

    if "noise_extracted" in st.session_state:

        st.image(
            st.session_state.noise_extracted,
            caption="Watermark Extracted After Gaussian Noise",
            use_container_width=False
        )