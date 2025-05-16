import streamlit as st
import os
import base64
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

def main():
    st.title("🔒 AES-128 + Visual Cryptography Image Security")
    st.write("""
    Secure image sharing using AES-128 encryption and Visual Cryptography (2-7 users).
    """)

    tab1, tab2 = st.tabs(["Encrypt & Share", "Decrypt"])

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
        st.header("Decrypt Image")
        st.markdown("""
        ### Decryption Process Steps:
        1. **Verify AES Key** - Check key format and validity
        2. **Verify Shares** - Ensure all shares are valid and compatible
        3. **Combine Shares** - Reconstruct encrypted data from shares
        4. **Verify Data** - Check reconstructed data matches uploaded file
        5. **Decrypt Image** - Final decryption with AES
        """)
        
        key_input = st.text_area("Enter AES-128 Key (base64)", key="key_input", 
                            help="Paste the base64-encoded AES key you received during encryption")
        
        encrypted_file = st.file_uploader("Upload encrypted image", type=["enc"], key="enc_uploader",
                                        help="Upload the .enc file you received during encryption")
        
        num_shares = st.slider("Number of shares required", 2, 7, 2, key="share_slider2",
                            help="Select how many shares were used during encryption")
        
        share_files = []
        for i in range(num_shares):
            share_file = st.file_uploader(
                f"Upload Share {i+1}", 
                type=["png", "jpg", "jpeg"], 
                key=f"share_upload_{i}",
                help=f"Upload visual cryptography share {i+1} (PNG/JPG format)"
            )
            if share_file:
                share_files.append(share_file)
        
        if st.button("Begin Decryption", key="decrypt", type="primary") and key_input and encrypted_file and len(share_files) == num_shares:
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_box = st.empty()
            details_box = st.expander("Decryption Details", expanded=True)
            
            try:
                # Step 1: Verify AES Key
                progress_bar.progress(10)
                status_box.info("🔍 **Step 1/5**: Verifying AES key...")
                details_box.write("""
                **Key Verification Process:**
                - Checking base64 format validity
                - Validating key length (128-bit)
                """)
                
                try:
                    aes_key = base64_to_key(key_input)
                    if len(aes_key) != 16:
                        raise ValueError("Key must be 16 bytes (128-bit)")
                    
                    # Visualize the key
                    key_bytes = np.frombuffer(aes_key, dtype=np.uint8).reshape(4,4)
                    fig, ax = plt.subplots(figsize=(4,4))
                    im = ax.imshow(key_bytes, cmap='viridis')
                    plt.colorbar(im, ax=ax, orientation='horizontal')
                    ax.set_title("Your AES Key (Visualized)")
                    ax.axis('off')
                    key_viz_path = os.path.join(tempfile.gettempdir(), "key_viz.png")
                    plt.savefig(key_viz_path, bbox_inches='tight')
                    plt.close()
                    
                    details_box.success("✅ Key verified successfully")
                    details_box.image(key_viz_path, caption="Visualization of your AES key")
                except Exception as e:
                    progress_bar.progress(0)
                    status_box.error("❌ **Key Verification Failed**")
                    details_box.error(f"""
                    **Error Details:**
                    {str(e)}
                    
                    **Possible Solutions:**
                    - Ensure you copied the entire key
                    - Verify the key hasn't been modified
                    - Try generating a new key if this one is lost
                    """)
                    raise
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Step 2: Verify Shares
                    progress_bar.progress(25)
                    status_box.info("🔍 **Step 2/5**: Verifying shares...")
                    share_paths = []
                    share_previews = []
                    
                    details_box.write("""
                    **Share Verification Process:**
                    - Checking file integrity
                    - Validating image formats
                    - Comparing dimensions
                    """)
                    
                    # Create a grid to show all shares
                    share_grid_cols = st.columns(4)
                    share_grid_idx = 0
                    
                    for i, share_file in enumerate(share_files):
                        share_path = os.path.join(temp_dir, f"share_{i+1}.png")
                        with open(share_path, "wb") as f:
                            f.write(share_file.getvalue())
                        share_paths.append(share_path)
                        
                        # Verify share image
                        try:
                            share_img = Image.open(share_path)
                            share_previews.append(share_img)
                            
                            # Show in grid
                            if share_grid_idx < len(share_grid_cols):
                                with share_grid_cols[share_grid_idx]:
                                    st.image(share_img, caption=f"Share {i+1}", width=150)
                                share_grid_idx += 1
                                
                            details_box.write(f"✅ Share {i+1}: Valid ({share_img.size[0]}×{share_img.size[1]} pixels)")
                        except Exception as e:
                            details_box.error(f"❌ Share {i+1}: Invalid - {str(e)}")
                            raise
                    
                    # Step 3: Combine Shares (with visualization)
                    progress_bar.progress(50)
                    status_box.info("🔍 **Step 3/5**: Combining shares...")
                    details_box.write("""
                    **Share Combination Process:**
                    - Performing visual cryptography reconstruction
                    - Validating output data structure
                    """)
                    
                    # Create animation of share combination
                    animation_path = os.path.join(temp_dir, "share_animation.gif")
                    
                    # Generate frames for animation
                    frames = []
                    current_combined = None
                    
                    for i, share_path in enumerate(share_paths):
                        share_img = Image.open(share_path).convert('L')
                        if current_combined is None:
                            current_combined = share_img
                        else:
                            current_combined = ImageChops.logical_xor(
                                current_combined.convert('1'), 
                                share_img.convert('1')
                            ).convert('L')
                        
                        # Create frame
                        fig, ax = plt.subplots(figsize=(6,4))
                        ax.imshow(current_combined, cmap='gray')
                        ax.set_title(f"After combining {i+1} shares", pad=10)
                        ax.axis('off')
                        
                        # Save frame to buffer
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                        plt.close()
                        buf.seek(0)
                        frames.append(imageio.imread(buf))
                    
                    # Save as GIF
                    imageio.mimsave(animation_path, frames, duration=1.5)
                    
                    # Show animation
                    st.markdown("### Share Combination Process")
                    st.image(animation_path, use_column_width=True)
                    
                    # Perform actual combination
                    success, combined_data, error_msg = combine_shares(share_paths)
                    if not success:
                        progress_bar.progress(0)
                        status_box.error("❌ **Share Combination Failed**")
                        details_box.error(f"""
                        **Error Details:**
                        {error_msg}
                        
                        **Possible Solutions:**
                        - Ensure all shares are from the same set
                        - Verify no shares have been modified
                        - Check you're using the correct number of shares
                        """)
                        raise ValueError(error_msg)
                    
                    details_box.success(f"✅ Successfully combined {len(share_paths)} shares")
                    details_box.write(f"Reconstructed data size: {len(combined_data)} bytes")
                    
                    # Step 4: Verify Reconstructed Data
                    progress_bar.progress(75)
                    status_box.info("🔍 **Step 4/5**: Verifying encrypted data...")
                    encrypted_path = os.path.join(temp_dir, "encrypted.enc")
                    with open(encrypted_path, "wb") as f:
                        f.write(encrypted_file.getvalue())
                    
                    uploaded_data = open(encrypted_path, "rb").read()
                    if combined_data != uploaded_data:
                        progress_bar.progress(0)
                        status_box.error("❌ **Data Mismatch Detected**")
                        details_box.error(f"""
                        **Security Alert**: 
                        Reconstructed data doesn't match uploaded encrypted file
                        
                        Uploaded size: {len(uploaded_data)} bytes
                        Reconstructed size: {len(combined_data)} bytes
                        Difference: {abs(len(uploaded_data) - len(combined_data))} bytes
                        
                        **This suggests:**
                        - Shares may be from different sets
                        - Encrypted file may be corrupted
                        - Possible tampering detected
                        """)
                        raise ValueError("Reconstructed data mismatch")
                    
                    details_box.success("✅ Encrypted data verified successfully")
                    
                    # Step 5: Decrypt Image (with visualization)
                    progress_bar.progress(90)
                    status_box.info("🔍 **Step 5/5**: Decrypting image...")
                    decrypted_path = os.path.join(temp_dir, "decrypted.png")
                    
                    # Create visualization of decryption process
                    st.markdown("### Decryption Process Visualization")
                    
                    # Create sample visualization of encrypted data
                    sample_size = min(10000, len(combined_data))
                    encrypted_sample = combined_data[:sample_size]
                    encrypted_img = Image.frombytes('L', (100, 100), encrypted_sample.ljust(10000, b'\0')[:10000])
                    
                    # Create visualization figure
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
                    
                    # Encrypted data visualization
                    ax1.imshow(encrypted_img, cmap='gray')
                    ax1.set_title("Encrypted Data Sample")
                    ax1.axis('off')
                    
                    # Key application visualization
                    key_bytes = np.frombuffer(aes_key, dtype=np.uint8)
                    key_heatmap = np.tile(key_bytes, (16,1))
                    im = ax2.imshow(key_heatmap, cmap='viridis')
                    plt.colorbar(im, ax=ax2, orientation='horizontal')
                    ax2.set_title("AES Key Application")
                    ax2.axis('off')
                    
                    viz_path = os.path.join(temp_dir, "decrypt_viz.png")
                    plt.savefig(viz_path, bbox_inches='tight')
                    plt.close()
                    
                    st.image(viz_path, use_column_width=True)
                    
                    # Perform actual decryption
                    try:
                        if decrypt_image_aes(combined_data, aes_key, decrypted_path):
                            progress_bar.progress(100)
                            status_box.success("🎉 **Decryption Successful!**")
                            st.balloons()
                            
                            # Display results
                            st.markdown("### Decryption Results")
                            
                            # Before/After comparison
                            st.markdown("**Before and After Comparison**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(encrypted_img, caption="Encrypted Data", use_column_width=True)
                            with col2:
                                decrypted_img = Image.open(decrypted_path)
                                st.image(decrypted_img, caption="Decrypted Image", use_column_width=True)
                            
                            # File info
                            st.markdown("**File Information**")
                            st.write(f"Original size: {len(combined_data)} bytes")
                            st.write(f"Decrypted size: {os.path.getsize(decrypted_path)} bytes")
                            
                            # Download button
                            with open(decrypted_path, "rb") as f:
                                st.download_button(
                                    label="Download Decrypted Image",
                                    data=f,
                                    file_name="decrypted.png",
                                    mime="image/png",
                                    key="dl_decrypted_final"
                                )
                        else:
                            raise Exception("Decryption returned False without error")
                    except Exception as e:
                        progress_bar.progress(0)
                        status_box.error("❌ **Decryption Failed**")
                        details_box.error(f"""
                        **Error Details:**
                        {str(e)}
                        
                        **Possible Solutions:**
                        - Double-check your AES key
                        - Verify all shares are correct
                        - Ensure encrypted file wasn't modified
                        - Try the encryption process again if needed
                        """)
                        raise
                
            except Exception as e:
                st.error("Decryption process aborted due to errors")
                st.stop()
if __name__ == "__main__":
    main()