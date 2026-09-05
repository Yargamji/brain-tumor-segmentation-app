import streamlit as st
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import tempfile
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import time
import tensorflow as tf

# Force CPU-only operation to avoid CUDA errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set page config
st.set_page_config(page_title="Glioma Segmentation", layout="wide")

# Initialize scaler
scaler = MinMaxScaler()

# Constants - model is bundled directly in the repo, no download needed
MODEL_PATH = os.path.join("saved_model", "3D_unet_100_epochs_2_batch_patch_training.keras")

# Model expects (96, 96, 96, 4) input
TARGET_SHAPE = (96, 96, 96, 4)


@st.cache_resource
def load_segmentation_model():
    """Load the model directly from the repo - no download step needed."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. "
                  "Make sure saved_model/ was included in your GitHub repo.")
        return None
    try:
        tf.get_logger().setLevel('ERROR')
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def process_uploaded_files(uploaded_files):
    """Load and scale each uploaded NIfTI file, sorting into modalities by filename."""
    modalities = {}

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name.lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        try:
            img = nib.load(tmp_path)
            img_data = img.get_fdata()

            img_data = scaler.fit_transform(
                img_data.reshape(-1, img_data.shape[-1])
            ).reshape(img_data.shape)

            if 't1n' in file_name:
                modalities['t1n'] = img_data
            elif 't1c' in file_name:
                modalities['t1c'] = img_data
            elif 't2f' in file_name:
                modalities['t2f'] = img_data
            elif 't2w' in file_name:
                modalities['t2w'] = img_data
            elif 'seg' in file_name:
                modalities['mask'] = img_data.astype(np.uint8)

        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return modalities


def extract_center_patch(volume, patch_size):
    """Extract the center 96x96x96 patch - matches how the model was trained
    (see Phase 3's extract_center_patch), NOT downsampling."""
    start_x = (volume.shape[0] - patch_size[0]) // 2
    start_y = (volume.shape[1] - patch_size[1]) // 2
    start_z = (volume.shape[2] - patch_size[2]) // 2
    patch = volume[
        start_x:start_x + patch_size[0],
        start_y:start_y + patch_size[1],
        start_z:start_z + patch_size[2]
    ]
    return patch, (start_x, start_y, start_z)


def prepare_input(modalities, patch_size=(96, 96, 96)):
    """Combine the 4 modalities, crop to 128^3, then extract the center 96^3 patch
    the model actually expects."""
    required = ['t1n', 't1c', 't2f', 't2w']
    if not all(m in modalities for m in required):
        return None, None, None, None

    combined = np.stack([
        modalities['t1n'],
        modalities['t1c'],
        modalities['t2f'],
        modalities['t2w']
    ], axis=3)

    # Crop to the same 128x128x128x4 region used during training/preprocessing
    combined = combined[56:184, 56:184, 13:141, :]
    full_shape = combined.shape

    # Extract the center 96x96x96 patch - this is what the model was trained on
    patch, offset = extract_center_patch(combined, patch_size)

    return patch, full_shape, combined, offset


def make_prediction(model, input_data):
    """Run the model on a single preprocessed 96^3 patch and return the predicted class map."""
    input_data = np.expand_dims(input_data, axis=0)
    prediction = model.predict(input_data, verbose=0)
    prediction_argmax = np.argmax(prediction, axis=4)[0, :, :, :]
    return prediction_argmax


def place_prediction_in_volume(prediction, offset, full_shape, patch_size=(96, 96, 96)):
    """Place the 96^3 predicted patch back into a full-size (128^3) volume at the
    same center offset it was extracted from. Everywhere outside the patch stays
    background (class 0), matching how the tutorial's own visualization handles this."""
    start_x, start_y, start_z = offset
    full_pred = np.zeros(full_shape[:3], dtype=prediction.dtype)
    full_pred[
        start_x:start_x + patch_size[0],
        start_y:start_y + patch_size[1],
        start_z:start_z + patch_size[2]
    ] = prediction
    return full_pred


def visualize_results(original_data, prediction, ground_truth=None):
    """Build a matplotlib figure comparing input, prediction, and (optionally) ground truth."""
    image_data = original_data[:, :, :, 1]  # T1c channel for display
    slice_indices = [50, 75, 90]

    fig, axes = plt.subplots(
        3, 3 if ground_truth is not None else 2,
        figsize=(10, 6)
    )

    for i, slice_idx in enumerate(slice_indices):
        img_slice = np.rot90(image_data[:, :, slice_idx])
        pred_slice = np.rot90(prediction[:, :, slice_idx])

        axes[i, 0].imshow(img_slice, cmap='gray')
        axes[i, 0].set_title(f'Input Image - Slice {slice_idx}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(pred_slice)
        axes[i, 1].set_title(f'Prediction - Slice {slice_idx}')
        axes[i, 1].axis('off')

        if ground_truth is not None:
            gt_slice = np.rot90(ground_truth[:, :, slice_idx])
            axes[i, 2].imshow(gt_slice)
            axes[i, 2].set_title(f'Ground Truth - Slice {slice_idx}')
            axes[i, 2].axis('off')

    plt.tight_layout()
    return fig


def main():
    st.title("3D Glioma Segmentation with U-Net")
    st.write("Upload MRI scans in NIfTI format for glioma segmentation")

    with st.expander("How to use this app"):
        st.markdown("""
        1. Upload **all four MRI modalities** (T1n, T1c, T2f, T2w) as NIfTI files (.nii.gz)
        2. Optionally upload a segmentation mask for comparison (must contain 'seg' in filename)
        3. Click 'Process and Predict'
        4. View the segmentation results and download the output

        **Note:** This model runs on CPU and may take a minute or two per prediction.
        """)

    model = load_segmentation_model()

    if model is None:
        st.error("Failed to load model. Please check the error message above.")
        return

    uploaded_files = st.file_uploader(
        "Upload MRI scans (NIfTI format)",
        type=['nii', 'nii.gz'],
        accept_multiple_files=True
    )

    if uploaded_files and len(uploaded_files) >= 4:
        if st.button("Process and Predict"):
            with st.spinner("Processing files..."):
                modalities = process_uploaded_files(uploaded_files)
                input_patch, full_shape, original_data, offset = prepare_input(modalities)

                if input_patch is None:
                    st.error("Could not prepare input data. Please ensure you've "
                              "uploaded all required modalities.")
                    return

                ground_truth = modalities.get('mask', None)
                if ground_truth is not None:
                    ground_truth = ground_truth[56:184, 56:184, 13:141]
                    ground_truth[ground_truth == 4] = 3

            with st.spinner("Making prediction (this may take a minute on CPU)..."):
                start_time = time.time()
                patch_prediction = make_prediction(model, input_patch)
                prediction = place_prediction_in_volume(patch_prediction, offset, full_shape)
                prediction = prediction.astype(np.int32)
                elapsed_time = time.time() - start_time

            st.success(f"Prediction completed in {elapsed_time:.2f} seconds")

            fig = visualize_results(original_data, prediction, ground_truth)
            st.pyplot(fig)

            st.subheader("Download Prediction")
            fd, tmp_path = tempfile.mkstemp(suffix='.nii.gz')
            os.close(fd)  # close the handle immediately so nib.save can write on Windows
            pred_img = nib.Nifti1Image(prediction, affine=np.eye(4), dtype=np.int32)
            nib.save(pred_img, tmp_path)
            with open(tmp_path, 'rb') as f:
                pred_data = f.read()
            os.unlink(tmp_path)

            st.download_button(
                label="Download Segmentation (NIfTI)",
                data=pred_data,
                file_name="glioma_segmentation.nii.gz",
                mime="application/octet-stream"
            )
    elif uploaded_files and len(uploaded_files) < 4:
        st.warning("Please upload all four modalities (T1n, T1c, T2f, T2w)")


if __name__ == "__main__":
    main()
