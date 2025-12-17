
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_model():
    import warnings
    warnings.filterwarnings('ignore')
    # Load model without compiling to avoid compatibility issues
    model = tf.keras.models.load_model("model/xray_model.hdf5", compile=False)
    # Recompile with current Keras defaults
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

class_names = ["NORMAL", "PNEUMONIA"]

st.write("""
         # Pneumonia Identification System
         """
         )

file = st.file_uploader("Please upload a chest scan file", type=["jpg","jpeg", "png"])

def import_and_predict(image_data, model):
    size = (180, 180)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image = np.asarray(image)
    
    # Ensure image has 3 channels (RGB)
    if len(image.shape) == 2:  # Grayscale image
        image = np.stack([image] * 3, axis=-1)  # Convert to RGB
    elif image.shape[2] == 4:  # RGBA image
        image = image[:, :, :3]  # Drop alpha channel
    
    img_reshape = image[np.newaxis, ...]  # Add batch dimension
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, width='stretch')
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0]) 
    
    # Calculate prediction and confidence
    class_index = np.argmax(score)
    confidence = 100 * np.max(score)
    
    # Display results
    st.subheader("📊 Prediction Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Class", class_names[class_index])
    with col2:
        st.metric("Confidence", f"{confidence:.2f}%")
    
    # Detailed scores
    with st.expander("🔬 Detailed Model Output", expanded=True):
        st.write("### Raw Predictions (Model Output)")
        st.info(f"""
        **Raw logits from neural network:**
        - NORMAL: {predictions[0][0]:.6f}
        - PNEUMONIA: {predictions[0][1]:.6f}
        
        These are unbounded values - they can be any real number.
        """)
        
        st.write("### Softmax Scores (Probabilities)")
        st.info(f"""
        **Converted to probabilities (always sum to 1.0):**
        - NORMAL: {score[0]:.6f} ({score[0]*100:.2f}%)
        - PNEUMONIA: {score[1]:.6f} ({score[1]*100:.2f}%)
        
        **Sum: {score[0] + score[1]:.6f}** (should be 1.0)
        """)
        
        # Visual comparison
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔴 NORMAL (Raw)", f"{predictions[0][0]:.4f}")
        with col2:
            st.metric("🔴 NORMAL (Softmax)", f"{score[0]:.4f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🟠 PNEUMONIA (Raw)", f"{predictions[0][1]:.4f}")
        with col2:
            st.metric("🟠 PNEUMONIA (Softmax)", f"{score[1]:.4f}")
        
    # Show which one is selected
    st.success(f"✅ **Final Decision**: {class_names[class_index]} with {confidence:.2f}% confidence")
    
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[class_index], confidence)
    )
