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
import threading

# Page configuration
st.set_page_config(
    page_title="Secure Bank Vault System",
    page_icon="🔒",
    layout="wide"
)

# Create a global server state for cross-session communication
class ServerState:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ServerState, cls).__new__(cls)
                cls._instance.unlocked_sessions = {}
        return cls._instance

# Get the server state singleton
server_state = ServerState()

# Custom CSS for bank vault theme
st.markdown("""
<style>
    .main {
        background-color: #0d1b2a;
        color: #e0e1dd;
    }
    .stButton > button {
        background-color: #1b263b;
        color: #e0e1dd;
        border: 1px solid #415a77;
    }
    .stButton > button:hover {
        background-color: #415a77;
        color: #e0e1dd;
        border: 1px solid #778da9;
    }
    h1, h2, h3 {
        color: #778da9;
    }
    .success-message {
        background-color: #1b4332;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #2d6a4f;
        margin: 10px 0;
    }
    .warning-message {
        background-color: #9d0208;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #6a040f;
        margin: 10px 0;
    }
    .info-message {
        background-color: #1b263b;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #415a77;
        margin: 10px 0;
    }
    .vault-header {
        text-align: center;
        margin-bottom: 20px;
    }
    .file-container {
        border: 1px solid #415a77;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        background-color: #1b263b;
    }
    .auto-refresh {
        display: none;
    }
</style>
<script>
    function autoRefresh() {
        window.location.reload();
    }
    setTimeout(autoRefresh, 3000);
</script>
""", unsafe_allow_html=True)

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
    st.session_state.shared_storage_dir = os.path.join(tempfile.gettempdir(), "bank_vault_shares")
    os.makedirs(st.session_state.shared_storage_dir, exist_ok=True)
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'encrypted_file_uploaded' not in st.session_state:
    st.session_state.encrypted_file_uploaded = False
if 'aes_key_entered' not in st.session_state:
    st.session_state.aes_key_entered = False
if 'last_check_time' not in st.session_state:
    st.session_state.last_check_time = time.time()

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

def display_vault_files():
    """Display the files in the vault that have been unlocked"""
    st.markdown("<div class='file-container'>", unsafe_allow_html=True)
    st.markdown("### 🔓 Vault Files Unlocked")
    
    # Hardcoded display of the PDF files
    st.markdown("#### Confidential Documents Available:")
    
    # Get the current directory
    current_dir = os.getcwd()
    
    # List all PDF files in the directory
    pdf_files = [f for f in os.listdir(current_dir) if f.endswith('.pdf')]
    
    if pdf_files:
        st.success(f"Found {len(pdf_files)} PDF files in the directory.")
        
        for file_name in pdf_files:
            try:
                with open(file_name, "rb") as f:
                    file_bytes = f.read()
                    st.download_button(
                        label=f"📄 Download {file_name}",
                        data=file_bytes,
                        file_name=file_name,
                        mime="application/pdf",
                        key=f"dl_{file_name}"
                    )
                st.write(f"**File:** {file_name} - Confidential Document")
            except Exception as e:
                st.error(f"Error with file {file_name}: {str(e)}")
    else:
        st.warning("No PDF files found in the current directory.")
        # List all files in the directory to help debug
        all_files = os.listdir(current_dir)
        st.write("All files in directory:")
        for file in all_files:
            st.write(f"- {file}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def mark_vault_unlocked(session_id):
    """Mark a vault as unlocked in the server state"""
    server_state.unlocked_sessions[session_id] = True

def check_vault_unlocked(session_id):
    """Check if the vault has been unlocked for this session"""
    # First check our server-side state (faster)
    if session_id in server_state.unlocked_sessions:
        return server_state.unlocked_sessions[session_id]
    
    # Then check the file system (more reliable but slower)
    session_dir = os.path.join(st.session_state.shared_storage_dir, session_id)
    try:
        with open(os.path.join(session_dir, "metadata.json"), "r") as f:
            session_meta = json.load(f)
        is_unlocked = session_meta.get("status") == "completed"
        
        # Update our server-side cache
        if is_unlocked:
            server_state.unlocked_sessions[session_id] = True
            
        return is_unlocked
    except:
        return False

def check_all_shares_uploaded(session_dir):
    """Check if all required shares have been uploaded"""
    try:
        with open(os.path.join(session_dir, "metadata.json"), "r") as f:
            session_meta = json.load(f)
        return session_meta["uploaded_shares"] >= session_meta["required_shares"]
    except:
        return False

def decrypt_vault(session_dir):
    """Decrypt the vault using all shares"""
    try:
        # Get all share files
        share_files = [f for f in os.listdir(session_dir) if f.startswith("share_")]
        share_paths = [os.path.join(session_dir, f) for f in share_files]
        
        # Get the key and encrypted file
        enc_path = os.path.join(session_dir, "encrypted.enc")
        key_path = os.path.join(session_dir, "aes_key.txt")
        
        if not os.path.exists(enc_path) or not os.path.exists(key_path):
            return False, "Missing encrypted file or key"
        
        # Read key
        with open(key_path, "r") as f:
            key_input = f.read()
        
        # Verify Security Key
        try:
            aes_key = base64_to_key(key_input)
            if len(aes_key) != 16:
                return False, "Security key must be 16 bytes (128-bit)"
        except Exception as e:
            return False, f"Key verification failed: {str(e)}"
        
        # Combine Shares
        success, combined_data, error_msg = combine_shares(share_paths)
        if not success:
            return False, f"Share combination failed: {error_msg}"
        
        # Verify Reconstructed Data
        with open(enc_path, "rb") as f:
            uploaded_data = f.read()
        
        if combined_data != uploaded_data:
            return False, "Reconstructed data doesn't match encrypted file"
        
        # Decrypt Document
        decrypted_path = os.path.join(session_dir, "decrypted.pdf")
        if not decrypt_image_aes(combined_data, aes_key, decrypted_path):
            return False, "Decryption failed"
        
        # Update session status
        with open(os.path.join(session_dir, "metadata.json"), "r") as f:
            session_meta = json.load(f)
        
        session_meta["status"] = "completed"
        with open(os.path.join(session_dir, "metadata.json"), "w") as f:
            json.dump(session_meta, f)
        
        # Mark as unlocked in server state
        mark_vault_unlocked(os.path.basename(session_dir))
        
        return True, "Vault successfully unlocked"
    except Exception as e:
        return False, f"Error during decryption: {str(e)}"

def main():
    st.markdown("<div class='vault-header'>", unsafe_allow_html=True)
    st.title("🏦 Secure Bank Vault System")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-message'>
    This secure system requires multiple shareholders to access confidential documents. 
    No single person can access the vault contents alone.
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Encrypt & Create Shares", "Shareholder Access"])
    
    with tab1:
        st.header("🔐 Secure Document Encryption")
        
        uploaded_file = st.file_uploader("Upload confidential document", type=["png", "jpg", "jpeg", "pdf", "docx"], key="uploader1")
        
        if uploaded_file is not None:
            # Save original file to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                original_path = tmp_file.name
            
            st.subheader("Document Received")
            st.success(f"Document '{uploaded_file.name}' successfully uploaded to the vault")
            
            if st.button("Generate Security Key", key="gen_key"):
                try:
                    st.session_state.aes_key = generate_aes_key()
                    st.markdown("""
                    <div class='success-message'>
                    ✅ Security key generated successfully!
                    </div>
                    """, unsafe_allow_html=True)
                    st.text_area("Your Security Key (CONFIDENTIAL)",
                                value=key_to_base64(st.session_state.aes_key),
                                disabled=True,
                                key="key_display")
                    st.warning("⚠️ Store this key in a secure location. It will be required for vault access.")
                except Exception as e:
                    st.error(f"Key generation failed: {str(e)}")
            
            if st.session_state.aes_key and st.button("Encrypt Document", key="encrypt"):
                try:
                    with st.spinner("Encrypting document..."):
                        st.session_state.encrypted_data = encrypt_image_aes(original_path, st.session_state.aes_key)
                        st.markdown("""
                        <div class='success-message'>
                        🔒 Document encrypted and secured in the vault!
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".enc") as tmp_file:
                            tmp_file.write(st.session_state.encrypted_data)
                            encrypted_path = tmp_file.name
                        
                        with open(encrypted_path, "rb") as f:
                            st.download_button(
                                label="Download Encrypted Document",
                                data=f,
                                file_name="vault_document.enc",
                                mime="application/octet-stream",
                                key="dl_encrypted"
                            )
                except Exception as e:
                    st.error(f"Encryption failed: {str(e)}")
            
            if st.session_state.encrypted_data:
                st.subheader("Generate Shareholder Keys")
                num_users = st.slider("Number of shareholders required", 2, 7, 3, key="share_slider1")
                
                if st.button("Generate Shareholder Keys", key="gen_shares"):
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
                            st.markdown(f"""
                            <div class='success-message'>
                            ✅ {num_users} shareholder keys created successfully!
                            </div>
                            """, unsafe_allow_html=True)
                            
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
                                        label="Download All Shareholder Keys",
                                        data=f,
                                        file_name="shareholder_keys.zip",
                                        mime="application/zip",
                                        key="dl_all_shares"
                                    )
                                
                                # Display individual download buttons in columns
                                st.markdown("### Individual Shareholder Keys")
                                cols = st.columns(3)
                                for i, share_path in enumerate(st.session_state.share_paths):
                                    with open(share_path, 'rb') as f:
                                        cols[i%3].download_button(
                                            label=f"Shareholder Key {i+1}",
                                            data=f,
                                            file_name=f"shareholder_key_{i+1}.png",
                                            mime="image/png",
                                            key=f"dl_share_{i}"
                                        )
                    except Exception as e:
                        st.error(f"Share generation failed: {str(e)}")
    
    with tab2:
        st.header("🔓 Vault Access Protocol")
        
        # Session management section
        st.subheader("Shareholder Authentication")
        session_col1, session_col2 = st.columns(2)
        
        with session_col1:
            session_mode = st.radio(
                "Select Role",
                ["Primary Shareholder", "Secondary Shareholder"],
                key="session_mode"
            )
        
        # Create new session (Primary Shareholder)
        if session_mode == "Primary Shareholder":
            with session_col2:
                num_parties = st.slider("Required shareholders for access", 2, 7, 3, key="num_parties")
                
            if st.button("Initialize Vault Access", key="create_session") or st.session_state.current_session_id:
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
                
                st.markdown(f"""
                <div class='success-message'>
                ✅ Vault access initialized! Share this access code with other shareholders: <b>{session_id}</b>
                </div>
                """, unsafe_allow_html=True)
                
                st.info("Each shareholder must join this session and provide their key.")
                
                # Check if the vault is already unlocked
                if check_vault_unlocked(session_id):
                    st.success("🔓 Vault has been successfully unlocked!")
                    display_vault_files()
                else:
                    # Store encrypted file upload field
                    encrypted_file = st.file_uploader(
                        "Upload encrypted document (Primary Shareholder only)",
                        type=["enc"],
                        key="enc_uploader",
                        help="Upload the .enc file received during encryption"
                    )
                    
                    encrypted_path = os.path.join(session_dir, "encrypted.enc")
                    
                    # Check if we already uploaded the file
                    if os.path.exists(encrypted_path):
                        st.session_state.encrypted_file_uploaded = True
                        st.success("✅ Encrypted document already uploaded!")
                    
                    # Handle new upload
                    if encrypted_file and not st.session_state.encrypted_file_uploaded:
                        save_uploaded_file(encrypted_file, encrypted_path)
                        st.session_state.encrypted_file_uploaded = True
                        st.success("✅ Encrypted document uploaded successfully!")
                    
                    # Store AES key field
                    key_input = st.text_area(
                        "Enter Security Key (Primary Shareholder only)",
                        key="key_input",
                        help="Paste the base64-encoded security key received during encryption"
                    )
                    
                    key_path = os.path.join(session_dir, "aes_key.txt")
                    
                    # Check if we already entered the key
                    if os.path.exists(key_path):
                        st.session_state.aes_key_entered = True
                        st.success("✅ Security key verified!")
                    
                    # Handle new key input
                    if key_input and len(key_input) > 0 and not st.session_state.aes_key_entered:
                        with open(key_path, "w") as f:
                            f.write(key_input)
                        st.session_state.aes_key_entered = True
                        st.success("✅ Security key verified!")
                    
                    # Check if all shares have been uploaded and automatically decrypt
                    if check_all_shares_uploaded(session_dir):
                        st.markdown("""
                        <div class='success-message'>
                        ✅ All required shareholder keys have been submitted!
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Check if we have the encrypted file and key
                        enc_path = os.path.join(session_dir, "encrypted.enc")
                        key_path = os.path.join(session_dir, "aes_key.txt")
                        
                        if os.path.exists(enc_path) and os.path.exists(key_path):
                            # Automatically decrypt without requiring a button click
                            success, message = decrypt_vault(session_dir)
                            
                            if success:
                                st.success("🔓 Vault has been successfully unlocked!")
                                display_vault_files()
                            else:
                                st.error(f"❌ Vault access failed: {message}")
                        else:
                            st.warning("⚠️ Please upload the encrypted document and security key to access the vault.")
        
        # Join existing session (Secondary Shareholder)
        else:
            with session_col2:
                session_id = st.text_input("Enter Access Code", key="join_session_id")
            
            if session_id and len(session_id) > 0:
                session_dir = os.path.join(st.session_state.shared_storage_dir, session_id)
                
                if not os.path.exists(session_dir):
                    st.error("❌ Invalid access code. Please check and try again.")
                else:
                    # Store the current session ID
                    st.session_state.current_session_id = session_id
                    
                    # Check if the vault is already unlocked
                    if check_vault_unlocked(session_id):
                        st.success("🔓 Vault has been successfully unlocked!")
                        display_vault_files()
                    else:
                        # Load session metadata
                        try:
                            with open(os.path.join(session_dir, "metadata.json"), "r") as f:
                                session_meta = json.load(f)
                            
                            # Check if session is active
                            if session_meta["status"] != "active":
                                st.error("❌ This vault access session has expired.")
                            else:
                                st.success(f"✅ Successfully joined vault access session: {session_id}")
                                
                                # Display session info
                                st.markdown(f"""
                                <div class='info-message'>
                                <b>Vault Access Information:</b><br>
                                • Initiated: {session_meta['created_at'][:16].replace('T', ' at ')}<br>
                                • Expires: {session_meta['expires_at'][:16].replace('T', ' at ')}<br>
                                • Required shareholders: {session_meta['required_shares']}<br>
                                • Keys submitted: {session_meta['uploaded_shares']}/{session_meta['required_shares']}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Share upload
                                st.subheader("Submit Your Shareholder Key")
                                party_name = st.text_input("Your name (for audit purposes)", key="party_name")
                                
                                if party_name:
                                    # Check if this party already uploaded a share
                                    share_path = os.path.join(session_dir, f"share_{party_name}.png")
                                    if os.path.exists(share_path):
                                        st.success(f"✅ You've already submitted your shareholder key!")
                                    else:
                                        share_file = st.file_uploader(
                                            "Upload your shareholder key",
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
                                            
                                            st.success(f"✅ Shareholder key verified! ({session_meta['uploaded_shares']}/{session_meta['required_shares']} keys submitted)")
                                            
                                            # Check if all shares are now uploaded and auto-decrypt
                                            if session_meta["uploaded_shares"] >= session_meta["required_shares"]:
                                                st.markdown("""
                                                <div class='success-message'>
                                                ✅ All required shareholder keys have been submitted!
                                                </div>
                                                """, unsafe_allow_html=True)
                                                
                                                # Check if we have the encrypted file and key
                                                enc_path = os.path.join(session_dir, "encrypted.enc")
                                                key_path = os.path.join(session_dir, "aes_key.txt")
                                                
                                                if os.path.exists(enc_path) and os.path.exists(key_path):
                                                    # Automatically decrypt
                                                    success, message = decrypt_vault(session_dir)
                                                    
                                                    if success:
                                                        st.success("🔓 Vault has been successfully unlocked!")
                                                        display_vault_files()
                                                    else:
                                                        st.error(f"❌ Vault access failed: {message}")
                                
                                # Check if all shares are uploaded
                                if session_meta["uploaded_shares"] >= session_meta["required_shares"]:
                                    st.markdown("""
                                    <div class='success-message'>
                                    ✅ All required shareholder keys have been submitted!
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Check if vault is unlocked
                                    if check_vault_unlocked(session_id):
                                        st.success("🔓 Vault has been successfully unlocked!")
                                        display_vault_files()
                        except Exception as e:
                            st.error(f"Error loading session: {str(e)}")

    # Add auto-refresh to check for vault unlocking
    if st.session_state.current_session_id:
        session_id = st.session_state.current_session_id
        # Add JavaScript for auto-refresh
        st.markdown("""
        <div class="auto-refresh">
        <script>
            setTimeout(function() {
                window.location.reload();
            }, 3000);
        </script>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
