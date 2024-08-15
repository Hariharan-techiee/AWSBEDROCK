import boto3
import json
import base64
import io
import streamlit as st
from PIL import Image
import time

# Initialize AWS clients
s3_client = boto3.client('s3')
bedrock = boto3.client(service_name="bedrock-runtime")

# Define S3 bucket and folder
BUCKET_NAME = 'bedrockmine'
FOLDER_NAME = 'techpost/'

def generate_text(prompt_data):
    prompt_data=prompt_data+"in 200 words "
    payload = {
        "prompt": prompt_data,
        "maxTokens": 512,
        "temperature": 0.8,
        "topP": 0.8
    }
    body = json.dumps(payload)
    model_id = "ai21.j2-mid-v1"
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json",
    )
    response_body = json.loads(response.get("body").read())
    response_text = response_body.get("completions")[0].get("data").get("text")
    return response_text

def generate_image(prompt_data):
    # Improve prompt for clarity
    prompt_data = f"{prompt_data} with a high-quality, detailed image"

    # Define prompt template and payload
    prompt_template = [{"text": prompt_data, "weight": 1}]
    payload = {
        "text_prompts": prompt_template,
        "cfg_scale": 15,  # Increased scale for better adherence to the prompt
        "seed": 42,       # Fixed seed for reproducibility
        "steps": 75,      # Increased steps for higher quality
        "width": 448,    # Higher resolution
        "height": 448   # Higher resolution
    }
    
    # Invoke the model
    body = json.dumps(payload)
    model_id = "stability.stable-diffusion-xl-v1"
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json",
    )
    
    # Decode and return the image
    response_body = json.loads(response.get("body").read())
    artifact = response_body.get("artifacts")[0]
    image_encoded = artifact.get("base64").encode("utf-8")
    image_bytes = base64.b64decode(image_encoded)
    
    return image_bytes


def upload_to_s3(file_name, file_bytes):
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=f'{FOLDER_NAME}{file_name}',
        Body=file_bytes,
        ContentType='image/png' if file_name.endswith('.png') else 'text/plain'
    )

def fetch_from_s3():
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=FOLDER_NAME)
    if 'Contents' in response:
        files = [content['Key'] for content in response['Contents']]
        return files
    return []

# Streamlit layout and design
st.set_page_config(page_title="Tech Blogs Using LLM", page_icon=":rocket:", layout="wide")

# Sidebar
with st.sidebar:
    st.header("Input Prompts")
    text_prompt = st.text_area("Enter a prompt for generating text:", "")
    image_prompt = st.text_area("Enter a prompt for generating image:", "")
    st.markdown("---")

    if st.button("Generate"):
        if text_prompt and image_prompt:
            # Generate text and image
            generated_text = generate_text(text_prompt)
            generated_image_bytes = generate_image(image_prompt)

            # Create unique file timestamp
            timestamp = int(time.time())
            image_file_name = f"{timestamp}.png"
            text_file_name = f"{timestamp}.txt"

            # Save and upload image to S3
            upload_to_s3(image_file_name, generated_image_bytes)

            # Store text in S3
            upload_to_s3(text_file_name, generated_text.encode('utf-8'))

            # Save results in session state
            st.session_state['generated_text'] = generated_text
            st.session_state['generated_image'] = generated_image_bytes
            st.success("Generated content has been uploaded to S3.")

    if st.button("Show Gallery"):
        st.session_state['show_gallery'] = True

# Main content area
if 'show_gallery' in st.session_state and st.session_state['show_gallery']:
    # Show Gallery
    st.subheader("Gallery")
    files = fetch_from_s3()

    # Collect all image and text pairs
    image_files = [file for file in files if file.endswith('.png')]
    text_files = [file for file in files if file.endswith('.txt')]

    # Ensure each image has a corresponding text file
    for image_file in image_files:
        # Extract timestamp from image filename
        timestamp = image_file.split('.')[0]
        text_file = f"{timestamp}.txt"

        if text_file in text_files:
            # Fetch image from S3
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=image_file)
            image_bytes = response['Body'].read()
            image = Image.open(io.BytesIO(image_bytes))

            # Fetch associated text
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=text_file)
            text_content = response['Body'].read().decode('utf-8')

            # Display image and text side by side
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption=image_file, width=300)  # Adjust width as needed
            with col2:
                st.write(f"**{text_file}**")
                st.write(text_content)
        else:
            # Debug: No matching text file found
            st.write(f"No text file found for image: {image_file}")

    if st.button("Back to Main Page"):
        st.session_state['show_gallery'] = False
else:
    if 'generated_text' in st.session_state and 'generated_image' in st.session_state:
        # Display results side by side
        st.subheader("Generated Content")

        # Define two columns
        col1, col2 = st.columns([1, 2])  # Adjust ratio as needed

        with col1:
            st.image(Image.open(io.BytesIO(st.session_state['generated_image'])), caption='Generated Image', width=300)  # Adjust width as needed

        with col2:
            st.write(st.session_state['generated_text'])

    else:
        st.header("Tech Blogs Using LLM")
        st.write("Generate and explore tech blogs using advanced language models.")
