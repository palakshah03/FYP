import streamlit as st
import os
import base64
import uuid
import json
import time
from datetime import datetime, timedelta
from aes_utils import (
    generate_aes_key,
    encrypt_image_aes,
    decrypt_image_aes,
    key_to_base64,
    base64_to_key
)
from vc_utils import generate_shares, combine_shares
import tempfile
import shutil
from PIL import Image, ImageDraw, ImageChops
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import io

# Page configuration
st.set_page_config(
    page_title="AES-128 + Visual Cryptography",
    page_icon="🔒",
    layout="wide"
)

# Session state initialization
if 'aes_key' not in st.session_state:
    st.session_state.aes_key = None
if 'encrypted_data' not in st.session_state:
    st.session_state.encrypted_data = None
if 'share_paths' not in st.session_state:
    st.session_state.share_paths = []
if 'shares_dir' not in st.session_state:
    st.session_state.shares_dir = None
if 'shares_generated' not in st.session_state:
    st.session_state.shares_generated = False
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'shared_storage_dir' not in st.session_state:
    st.session_state.shared_storage_dir = os.path.join(tempfile.gettempdir(), "multi_party_shares")
    os.makedirs(st.session_state.shared_storage_dir, exist_ok=True)
if 'decryption_sessions' not in st.session_state:
    st.session_state.decryption_sessions = {}
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'encrypted_file_uploaded' not in st.session_state:
    st.session_state.encrypted_file_uploaded = False
if 'aes_key_entered' not in st.session_state:
    st.session_state.aes_key_entered = False

def display_image(image_path, caption):
    """Helper function to display images"""
    try:
        image = Image.open(image_path)
        st.image(image, caption=caption, use_column_width=True)
    except Exception as e:
        st.error(f"Error displaying image: {e}")

def create_share_animation(share_paths, output_path):
    """Create an animation showing share combination"""
    shares = [Image.open(path) for path in share_paths]
    width, height = shares[0].size
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.axis('off')
    plt.title("Share Combination Process", pad=20)
    
    # Initialize with first share
    combined = shares[0].copy()
    img_plot = ax.imshow(combined, cmap='gray')
    
    def update(frame):
        nonlocal combined
        if frame < len(shares)-1:
            # Show XOR operation between current combined and next share
            next_share = shares[frame+1]
            combined = ImageChops.logical_xor(combined.convert('1'), next_share.convert('1')).convert('L')
            img_plot.set_array(combined)
            ax.set_title(f"Combining Share {frame+2} of {len(shares)}", pad=20)
        return img_plot,
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(shares),
                         interval=1000, blit=True)
    
    # Save as GIF
    anim.save(output_path, writer='pillow', fps=1, dpi=100)
    plt.close()

def create_decryption_visualization(encrypted_img_path, key, output_path):
    """Create visualization of decryption process"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Load encrypted image
    encrypted_img = Image.open(encrypted_img_path)
    ax1.imshow(encrypted_img, cmap='gray')
    ax1.set_title("Encrypted Image")
    ax1.axis('off')
    
    # Create key visualization
    key_visual = np.frombuffer(key, dtype=np.uint8).reshape(4,4)
    key_img = ax2.imshow(key_visual, cmap='viridis')
    ax2.set_title("AES Key Application")
    ax2.axis('off')
    
    # Add colorbar for key
    plt.colorbar(key_img, ax=ax2, orientation='horizontal')
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_uploaded_file(uploaded_file, save_path):
    """Save an uploaded file to the specified path"""
    if uploaded_file is not None:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        return True
    return False

def main():
    st.title("🔒 AES-128 + Visual Cryptography Image Security")
    
    st.write("""
    Secure image sharing using AES-128 encryption and Visual Cryptography (2-7 users).
    """)
    
    tab1, tab2 = st.tabs(["Encrypt & Share", "Decrypt Image (Multi-Party)"])
    
    with tab1:
        st.header("Encrypt and Share Image")
        
        uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp"], key="uploader1")
        
        if uploaded_file is not None:
            # Save original image to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                original_path = tmp_file.name
            
            st.subheader("Original Image")
            display_image(original_path, "Original Image")
            
            if st.button("Generate AES-128 Key", key="gen_key"):
                try:
                    st.session_state.aes_key = generate_aes_key()
                    st.success("AES-128 Key generated!")
                    st.text_area("Your AES Key (save securely)",
                                value=key_to_base64(st.session_state.aes_key),
                                disabled=True,
                                key="key_display")
                except Exception as e:
                    st.error(f"Key generation failed: {str(e)}")
            
            if st.session_state.aes_key and st.button("Encrypt Image", key="encrypt"):
                try:
                    with st.spinner("Encrypting..."):
                        st.session_state.encrypted_data = encrypt_image_aes(original_path, st.session_state.aes_key)
                        st.success("Image encrypted!")
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".enc") as tmp_file:
                            tmp_file.write(st.session_state.encrypted_data)
                            encrypted_path = tmp_file.name
                        
                        with open(encrypted_path, "rb") as f:
                            st.download_button(
                                label="Download Encrypted Image",
                                data=f,
                                file_name="encrypted.enc",
                                mime="application/octet-stream",
                                key="dl_encrypted"
                            )
                except Exception as e:
                    st.error(f"Encryption failed: {str(e)}")
            
            if st.session_state.encrypted_data:
                st.subheader("Generate Shares")
                num_users = st.slider("Number of shares", 2, 7, 2, key="share_slider1")
                
                if st.button("Generate Shares", key="gen_shares"):
                    try:
                        # Save encrypted data to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".enc") as tmp_file:
                            tmp_file.write(st.session_state.encrypted_data)
                            encrypted_data_path = tmp_file.name
                        
                        # Create temp dir for shares
                        with tempfile.TemporaryDirectory() as temp_dir:
                            st.session_state.share_paths, st.session_state.shares_dir = generate_shares(
                                encrypted_data_path, num_users, temp_dir
                            )
                            
                            st.session_state.shares_generated = True
                            st.success(f"{num_users} shares created!")
                            
                            # Create a temporary directory for download
                            with tempfile.TemporaryDirectory() as download_dir:
                                # Copy all shares to the download directory
                                download_shares_dir = os.path.join(download_dir, "shares")
                                shutil.copytree(st.session_state.shares_dir, download_shares_dir)
                                
                                # Create a downloadable folder
                                shutil.make_archive(download_shares_dir, 'zip', download_shares_dir)
                                
                                # Download button for all shares
                                with open(f"{download_shares_dir}.zip", "rb") as f:
                                    st.download_button(
                                        label="Download All Shares (as folder)",
                                        data=f,
                                        file_name="shares.zip",
                                        mime="application/zip",
                                        key="dl_all_shares"
                                    )
                                
                                # Display individual download buttons in columns
                                cols = st.columns(3)
                                for i, share_path in enumerate(st.session_state.share_paths):
                                    with open(share_path, 'rb') as f:
                                        cols[i%3].download_button(
                                            label=f"Download Share {i+1}",
                                            data=f,
                                            file_name=f"share_{i+1}.png",
                                            mime="image/png",
                                            key=f"dl_share_{i}"
                                        )
                    except Exception as e:
                        st.error(f"Share generation failed: {str(e)}")
    
    with tab2:
        st.header("Decrypt Image (Multi-Party)")
        
        # Session management section
        st.subheader("Session Management")
        session_col1, session_col2 = st.columns(2)
        
        with session_col1:
            session_mode = st.radio(
                "Select Mode",
                ["Create New Session", "Join Existing Session"],
                key="session_mode"
            )
        
        # Create new session
        if session_mode == "Create New Session":
            with session_col2:
                num_parties = st.slider("Number of parties required", 2, 7, 2, key="num_parties")
                
            if st.button("Create Decryption Session", key="create_session") or st.session_state.current_session_id:
                # Generate a unique session ID if we don't have one
                if not st.session_state.current_session_id:
                    session_id = str(uuid.uuid4())[:8]
                    st.session_state.current_session_id = session_id
                    
                    # Create session directory
                    session_dir = os.path.join(st.session_state.shared_storage_dir, session_id)
                    os.makedirs(session_dir, exist_ok=True)
                    
                    # Store session metadata
                    session_meta = {
                        "created_at": datetime.now().isoformat(),
                        "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
                        "required_shares": num_parties,
                        "uploaded_shares": 0,
                        "creator": st.session_state.session_id,
                        "status": "active"
                    }
                    
                    with open(os.path.join(session_dir, "metadata.json"), "w") as f:
                        json.dump(session_meta, f)
                else:
                    session_id = st.session_state.current_session_id
                    session_dir = os.path.join(st.session_state.shared_storage_dir, session_id)
                    
                    # Load existing metadata
                    with open(os.path.join(session_dir, "metadata.json"), "r") as f:
                        session_meta = json.load(f)
                
                st.success(f"Session created! Share this code with all parties: **{session_id}**")
                st.info("Each party should join this session and upload their share.")
                
                # Store encrypted file upload field
                encrypted_file = st.file_uploader(
                    "Upload encrypted image (only session creator)",
                    type=["enc"],
                    key="enc_uploader",
                    help="Upload the .enc file received during encryption"
                )
                
                encrypted_path = os.path.join(session_dir, "encrypted.enc")
                
                # Check if we already uploaded the file
                if os.path.exists(encrypted_path):
                    st.session_state.encrypted_file_uploaded = True
                    st.success("Encrypted file already uploaded!")
                
                # Handle new upload
                if encrypted_file and not st.session_state.encrypted_file_uploaded:
                    save_uploaded_file(encrypted_file, encrypted_path)
                    st.session_state.encrypted_file_uploaded = True
                    st.success("Encrypted file uploaded successfully!")
                
                # Store AES key field
                key_input = st.text_area(
                    "Enter AES-128 Key (only session creator)",
                    key="key_input",
                    help="Paste the base64-encoded AES key received during encryption"
                )
                
                key_path = os.path.join(session_dir, "aes_key.txt")
                
                # Check if we already entered the key
                if os.path.exists(key_path):
                    st.session_state.aes_key_entered = True
                    st.success("AES key already stored!")
                
                # Handle new key input
                if key_input and len(key_input) > 0 and not st.session_state.aes_key_entered:
                    with open(key_path, "w") as f:
                        f.write(key_input)
                    st.session_state.aes_key_entered = True
                    st.success("AES key stored successfully!")
        
        # Join existing session
        else:
            with session_col2:
                session_id = st.text_input("Enter Session ID", key="join_session_id")
            
            if session_id and len(session_id) > 0:
                session_dir = os.path.join(st.session_state.shared_storage_dir, session_id)
                
                if not os.path.exists(session_dir):
                    st.error("Session not found. Please check the ID and try again.")
                else:
                    # Store the current session ID
                    st.session_state.current_session_id = session_id
                    
                    # Load session metadata
                    try:
                        with open(os.path.join(session_dir, "metadata.json"), "r") as f:
                            session_meta = json.load(f)
                        
                        # Check if session is active
                        if session_meta["status"] != "active":
                            st.error("This session is no longer active.")
                        else:
                            st.success(f"Joined session: {session_id}")
                            
                            # Display session info
                            st.info(f"""
                            **Session Information:**
                            - Created: {session_meta['created_at']}
                            - Expires: {session_meta['expires_at']}
                            - Required shares: {session_meta['required_shares']}
                            - Shares uploaded: {session_meta['uploaded_shares']}
                            """)
                            
                            # Share upload
                            st.subheader("Upload Your Share")
                            party_name = st.text_input("Your name/identifier", key="party_name")
                            
                            if party_name:
                                # Check if this party already uploaded a share
                                share_path = os.path.join(session_dir, f"share_{party_name}.png")
                                if os.path.exists(share_path):
                                    st.success(f"You've already uploaded your share as {party_name}!")
                                else:
                                    share_file = st.file_uploader(
                                        "Upload your share",
                                        type=["png", "jpg", "jpeg"],
                                        key=f"party_share_upload_{party_name}"
                                    )
                                    
                                    if share_file:
                                        # Save share to session directory
                                        save_uploaded_file(share_file, share_path)
                                        
                                        # Update metadata
                                        session_meta["uploaded_shares"] += 1
                                        with open(os.path.join(session_dir, "metadata.json"), "w") as f:
                                            json.dump(session_meta, f)
                                        
                                        st.success(f"Share uploaded successfully! ({session_meta['uploaded_shares']}/{session_meta['required_shares']} shares)")
                            
                            # Check if all shares are uploaded
                            if session_meta["uploaded_shares"] >= session_meta["required_shares"]:
                                st.success("All required shares have been uploaded!")
                                
                                # Check if we have the encrypted file and key
                                enc_path = os.path.join(session_dir, "encrypted.enc")
                                key_path = os.path.join(session_dir, "aes_key.txt")
                                
                                if os.path.exists(enc_path) and os.path.exists(key_path):
                                    if st.button("Begin Decryption", key="begin_multi_decrypt"):
                                        # Get all share files
                                        share_files = [f for f in os.listdir(session_dir) if f.startswith("share_")]
                                        share_paths = [os.path.join(session_dir, f) for f in share_files]
                                        
                                        # Read key
                                        with open(key_path, "r") as f:
                                            key_input = f.read()
                                        
                                        # Initialize progress tracking
                                        progress_bar = st.progress(0)
                                        status_box = st.empty()
                                        details_box = st.expander("Decryption Details", expanded=True)
                                        
                                        try:
                                            # Step 1: Verify AES Key
                                            progress_bar.progress(10)
                                            status_box.info("🔍 **Step 1/5**: Verifying AES key...")
                                            
                                            try:
                                                aes_key = base64_to_key(key_input)
                                                if len(aes_key) != 16:
                                                    raise ValueError("Key must be 16 bytes (128-bit)")
                                                details_box.success("✅ Key verified successfully")
                                            except Exception as e:
                                                progress_bar.progress(0)
                                                status_box.error("❌ **Key Verification Failed**")
                                                details_box.error(f"Error: {str(e)}")
                                                raise
                                            
                                            # Step 2: Verify Shares
                                            progress_bar.progress(25)
                                            status_box.info("🔍 **Step 2/5**: Verifying shares...")
                                            
                                            # Display share info
                                            details_box.write(f"Found {len(share_paths)} shares from different parties:")
                                            for i, path in enumerate(share_paths):
                                                party = os.path.basename(path).replace("share_", "").replace(".png", "")
                                                details_box.write(f"- Share from: {party}")
                                            
                                            # Step 3: Combine Shares
                                            progress_bar.progress(50)
                                            status_box.info("🔍 **Step 3/5**: Combining shares...")
                                            
                                            # Perform actual combination
                                            success, combined_data, error_msg = combine_shares(share_paths)
                                            
                                            if not success:
                                                progress_bar.progress(0)
                                                status_box.error("❌ **Share Combination Failed**")
                                                details_box.error(f"Error: {error_msg}")
                                                raise ValueError(error_msg)
                                            
                                            details_box.success(f"✅ Successfully combined {len(share_paths)} shares")
                                            
                                            # Step 4: Verify Reconstructed Data
                                            progress_bar.progress(75)
                                            status_box.info("🔍 **Step 4/5**: Verifying encrypted data...")
                                            
                                            with open(enc_path, "rb") as f:
                                                uploaded_data = f.read()
                                            
                                            if combined_data != uploaded_data:
                                                progress_bar.progress(0)
                                                status_box.error("❌ **Data Mismatch Detected**")
                                                details_box.error("Reconstructed data doesn't match uploaded encrypted file")
                                                raise ValueError("Reconstructed data mismatch")
                                            
                                            details_box.success("✅ Encrypted data verified successfully")
                                            
                                            # Step 5: Decrypt Image
                                            progress_bar.progress(90)
                                            status_box.info("🔍 **Step 5/5**: Decrypting image...")
                                            
                                            decrypted_path = os.path.join(session_dir, "decrypted.png")
                                            
                                            # Perform actual decryption
                                            if decrypt_image_aes(combined_data, aes_key, decrypted_path):
                                                progress_bar.progress(100)
                                                status_box.success("🎉 **Decryption Successful!**")
                                                st.balloons()
                                                
                                                # Display results
                                                st.markdown("### Decryption Results")
                                                decrypted_img = Image.open(decrypted_path)
                                                st.image(decrypted_img, caption="Decrypted Image", use_column_width=True)
                                                
                                                # Download button
                                                with open(decrypted_path, "rb") as f:
                                                    st.download_button(
                                                        label="Download Decrypted Image",
                                                        data=f,
                                                        file_name="decrypted.png",
                                                        mime="image/png",
                                                        key="dl_decrypted_multi"
                                                    )
                                                
                                                # Update session status
                                                session_meta["status"] = "completed"
                                                with open(os.path.join(session_dir, "metadata.json"), "w") as f:
                                                    json.dump(session_meta, f)
                                            else:
                                                raise Exception("Decryption returned False without error")
                                        
                                        except Exception as e:
                                            progress_bar.progress(0)
                                            status_box.error("❌ **Decryption Failed**")
                                            details_box.error(f"Error: {str(e)}")
                                            st.error("Decryption process aborted due to errors")
                                else:
                                    st.warning("Waiting for session creator to upload encrypted file and AES key.")
                    except Exception as e:
                        st.error(f"Error loading session: {str(e)}")

if __name__ == "__main__":
    main()
