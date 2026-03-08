import cv2
import streamlit as st
import numpy as np
import dark_channel_prior as dcp
import inference as inf

# Function to remove noise from an image
def remove_noise(image):
    try:
        processed_image, alpha_map = dcp.haze_removal(image, w_size=15, a_omega=0.95, gf_w_size=200, eps=1e-6)
        return processed_image
    except Exception as e:
        st.error(f"Error in noise removal: {e}")
        return image  # Return original image if error occurs

# Function to perform object detection on an image
def detect_objects(image):
    try:
        output_image, class_names = inf.detect(image)
        return output_image, class_names
    except Exception as e:
        st.error(f"Error in object detection: {e}")
        return image, []  # Return original image if error occurs

# Main function for Streamlit app
def app():
    st.title("Ocean Waste Detection using YOLO")
    st.text("Upload an image to detect objects")

    # Allow user to upload an image
    file = st.file_uploader("Choose file", type=["jpg", "jpeg", "png"])

    if file is not None:
        st.text("Uploading image...")
        
        # Read the image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        input_image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        if input_image is None:
            st.error("Error loading image. Please try again.")
            return
        
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (416, 416))

        st.text("Input Image:")
        st.image(input_image)

        # Process the input
        st.text("Removing noise from input...")
        processed_image = remove_noise(input_image)
        st.image(processed_image)

        # Run the model
        st.text("Running the model...")
        output_image, class_names = detect_objects(processed_image)

        # Display the output
        st.text("Output Image:")
        st.image(output_image)

        if len(class_names) == 0:
            st.success("The water is clear!!!")
        else:
            st.error(f"Waste Detected!!!\nThe image has {class_names}")

# Run the app
if __name__ == "__main__":
    app()
