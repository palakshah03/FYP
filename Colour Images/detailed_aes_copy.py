import streamlit as st
import numpy as np
from PIL import Image
import base64
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import new as Random
from base64 import b64encode, b64decode
import io
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class AESCipher:
    def __init__(self, data, key):
        self.block_size = 16
        self.data = data
        self.key = hashlib.sha256(key.encode()).digest()[:32]
        self.pad = lambda s: s + (self.block_size - len(s) % self.block_size) * chr(self.block_size - len(s) % self.block_size)
        self.unpad = lambda s: s[:-ord(s[len(s) - 1:])]

    def encrypt(self):
        plain_text = self.pad(self.data)
        iv = Random().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_OFB, iv)
        return b64encode(iv + cipher.encrypt(plain_text.encode())).decode()

    def decrypt(self):
        cipher_text = b64decode(self.data.encode())
        iv = cipher_text[:self.block_size]
        cipher = AES.new(self.key, AES.MODE_OFB, iv)
        return self.unpad(cipher.decrypt(cipher_text[self.block_size:])).decode()

class HybridCryptography:
    def __init__(self, key):
        self.key = key
        self.hashed_key = hashlib.sha256(key.encode()).hexdigest()

    def generate_visual_key(self, height, width):
        h = len(self.key)
        C = np.ones((h, width, 1), dtype='uint8')
        
        for i in range(h):
            j = ord(self.key[i])
            for k in range(width):
                if k < j:
                    C[i][k][0] = 0
                else:
                    break
                    
        R = np.ones((h, width, 3), dtype='uint8')
        P = np.ones((h, width, 3), dtype='uint8')
        
        for i in range(h):
            for j in range(width):
                r = np.random.normal(0, 1, 1)
                R[i][j][0] = r
                P[i][j][0] = R[i][j][0] ^ C[i][j][0]
                
        return R, P

    def encrypt(self, input_image, share_size=2):
        # Convert image to base64
        img_byte_arr = input_image.tobytes()
        img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
        
        # AES encryption
        aes_cipher = AESCipher(img_base64, self.hashed_key)
        aes_encrypted = aes_cipher.encrypt()
        
        # Generate visual shares
        image_array = np.array(input_image)
        (row, column, depth) = image_array.shape
        shares = np.random.randint(0, 256, size=(row, column, depth, share_size))
        shares[:,:,:,-1] = image_array.copy()
        
        for i in range(share_size-1):
            shares[:,:,:,-1] = shares[:,:,:,-1] ^ shares[:,:,:,i]
        
        # Generate visual key
        R, P = self.generate_visual_key(row, 255)
        
        return {
            'aes_cipher': aes_encrypted,
            'visual_shares': shares,
            'key_share_r': R,
            'key_share_p': P
        }

    def decrypt(self, encrypted_data, show_steps=False):
        if show_steps:
            st.write("Step 1: Reconstructing key from key shares...")
        
        R = encrypted_data['key_share_r']
        P = encrypted_data['key_share_p']
        h, w = R.shape[:2]
        
        CK = np.ones((h, w, 1), dtype='uint8')
        for i in range(h):
            for j in range(w):
                CK[i][j][0] = P[i][j][0] ^ R[i][j][0]
        
        if show_steps:
            st.image(CK, caption="Reconstructed Key Pattern", use_column_width=True)
            st.write("Step 2: Extracting original key...")
        
        reconstructed_key = []
        for i in range(len(CK)):
            count = 0
            for j in range(len(CK[i])):
                if CK[i][j][0] == 0:
                    count += 1
            reconstructed_key.append(chr(count))
        
        reconstructed_key = "".join(reconstructed_key)
        
        if show_steps:
            st.write("Step 3: Combining visual shares...")
        
        shares = encrypted_data['visual_shares']
        (row, column, depth, share_size) = shares.shape
        shares_image = shares.copy()
        
        # Show intermediate combinations if requested
        if show_steps:
            for i in range(share_size-1):
                shares_image[:,:,:,-1] = shares_image[:,:,:,-1] ^ shares_image[:,:,:,i]
                st.image(shares_image[:,:,:,-1], 
                        caption=f"After combining share {i+1}", 
                        use_column_width=True)
        else:
            for i in range(share_size-1):
                shares_image[:,:,:,-1] = shares_image[:,:,:,-1] ^ shares_image[:,:,:,i]
        
        if show_steps:
            st.write("Step 4: Decrypting AES cipher...")
        
        decrypted_base64 = AESCipher(encrypted_data['aes_cipher'], 
                                    hashlib.sha256(reconstructed_key.encode()).hexdigest()).decrypt()
        
        img_bytes = base64.b64decode(decrypted_base64.encode('utf-8'))
        final_image = Image.frombytes('RGB', (column, row), img_bytes)
        
        if show_steps:
            st.write("Step 5: Final decrypted image")
            st.image(final_image, caption="Decrypted Image", use_column_width=True)
        
        return final_image

def main():
    st.set_page_config(page_title="Hybrid Cryptography System", layout="wide")

    st.title("🔒 Interactive Hybrid Cryptography Lecture Demo")
    st.markdown("## AES-256 + Visual Cryptography Educational Toolkit")
    st.markdown("---")

    # Initialize session state with clear documentation
    if 'processed' not in st.session_state:
        st.session_state.update({
            'processed': False,        # Flag for completion status
            'encrypted_data': None,    # Stores AES cipher + visual shares
            'decrypted_image': None,   # Final decrypted image
            'current_step': 1,         # Guided workflow step
            'input_image': None        # Original image storage
        })

    # Sidebar controls with enhanced tooltips
    with st.sidebar:
        st.header("🎛️ Control Panel")
        with st.expander("📖 Workflow Overview", expanded=True):
            st.markdown("""
            **Cryptography Process Flow:**
            1. **Image Upload**: Input RGB image
            2. **AES-256 Encryption**: 
               - Key derivation via SHA-256
               - OFB mode encryption
            3. **Visual Cryptography**:
               - Split into N shares
               - Generate key patterns (R/P)
            4. **Decryption**:
               - Share combination
               - AES key reconstruction
               - Image recovery
            """)

        uploaded_file = st.file_uploader("📤 Upload Image", type=['png', 'jpg', 'jpeg'],
                                       help="Supported formats: PNG, JPG, JPEG (max 5MB)")
        key = st.text_input("🔑 Encryption Key", type="password",
                          help="Minimum 8 characters - securely hashed using SHA-256")
        share_size = st.slider("🔢 Number of Shares", 2, 8, 4,
                             help="Number of visual shares to generate (k-of-n scheme)")
        process_btn = st.button("🚀 Start Process", 
                              disabled=not (uploaded_file and len(key)>=8),
                              key="process_btn")

    # Main processing workflow
    if process_btn and uploaded_file and key:
        try:
            with st.spinner("🔨 Processing - Generating AES key and visual shares..."):
                # Initialize cryptography system with detailed error handling
                hybrid_crypto = HybridCryptography(key)
                
                # Convert and validate input image
                input_image = Image.open(uploaded_file).convert('RGB')
                if input_image.size[0] * input_image.size[1] > 10_000_000:
                    raise ValueError("Image resolution too high (max 10MP)")
                
                # Execute full encryption pipeline
                encrypted_data = hybrid_crypto.encrypt(input_image, share_size)
                decrypted_image = hybrid_crypto.decrypt(encrypted_data)

                # Update session state
                st.session_state.update({
                    'processed': True,
                    'encrypted_data': encrypted_data,
                    'decrypted_image': decrypted_image,
                    'input_image': input_image,
                    'current_step': 1
                })
                st.rerun()

        except Exception as e:
            st.error(f"🚨 Processing Error: {str(e)}")
            st.error("Possible issues: Invalid image format, memory limits, or crypto operations failure")

    # Main visualization pipeline
    if st.session_state.processed:
        st.markdown("## Cryptographic Process Visualization")
        
        # Step 1: AES Encryption Visualization
        with st.container(border=True):
            st.markdown(f"### 1️⃣ Step {st.session_state.current_step}: AES-256 Encryption")
            cols = st.columns([2, 3])
            
            with cols[0]:
                st.image(st.session_state.input_image,
                       caption="Original Image",
                       use_column_width=True)

            with cols[1]:
                # Key derivation visualization
                key_hash = hashlib.sha256(key.encode()).digest()
                key_array = np.frombuffer(key_hash[:32], dtype=np.uint8)  # Use first 32 bytes
                
                fig, ax = plt.subplots(figsize=(8, 4))
                im = ax.imshow(key_array.reshape(4, 8), cmap='viridis', aspect='auto')
                plt.colorbar(im, ax=ax, label='Byte Value')
                ax.set_title("AES-256 Key Matrix (4x8 bytes)")
                ax.set_xticks([])
                ax.set_yticks([])
                st.pyplot(fig)

                # Cryptographic details expander
                with st.expander("🔍 AES-256 Technical Details"):
                    st.markdown("""
                    **Key Derivation Function:**
                    ```python
                    hashed_key = hashlib.sha256(user_key.encode()).digest()[:32]
                    ```
                    
                    **Encryption Parameters:**
                    - Mode: Output Feedback (OFB)
                    - Initialization Vector: 16 random bytes
                    - Block Size: 128 bits
                    - Padding: PKCS#7
                    
                    **Mathematical Foundation:**
                    ```python
                    cipher = AES.new(
                        key, 
                        AES.MODE_OFB,
                        iv=iv
                    )
                    ciphertext = iv + cipher.encrypt(padded_data)
                    ```
                    """)

            if st.button("Next Step →", key="step1_next"):
                st.session_state.current_step = 2
                st.rerun()

        # Step 2: Visual Cryptography Visualization
        if st.session_state.current_step >= 2:
            st.markdown("<div style='text-align: center; font-size: 30px; margin: 20px 0'>⬇️</div>", 
                      unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown(f"### 2️⃣ Step {st.session_state.current_step}: Visual Cryptography Shares")
                
                # Share gallery with inspection
                cols = st.columns(4)
                for idx in range(share_size):
                    with cols[idx % 4]:
                        share_img = Image.fromarray(
                            st.session_state.encrypted_data['visual_shares'][:,:,:,idx].astype(np.uint8)
                        )
                        st.image(share_img, 
                               caption=f"Share #{idx+1}", 
                               use_column_width=True)
                        
                        # Pixel analysis on hover
                        with st.expander(f"Share {idx+1} Analysis"):
                            st.image(share_img, use_column_width=True)
                            st.markdown(f"""
                            **Statistical Analysis:**
                            - Mean Intensity: {np.mean(share_img):.2f}
                            - Entropy: {skimage.measure.shannon_entropy(share_img):.2f}
                            - Unique Colors: {len(np.unique(share_img.reshape(-1, 3), axis=0))}
                            """)

                # Visual cryptography explanation
                with st.expander("📚 Visual Cryptography Theory"):
                    st.markdown("""
                    **XOR-based Secret Sharing:**
                    ```python
                    final_share = original_image ^ share1 ^ share2 ^ ... ^ shareN-1
                    ```
                    
                    **Key Share Generation:**
                    - R Matrix: Random binary pattern
                    - P Matrix: Secret pattern (R ⊕ C)
                    - C Matrix: Key-derived control matrix
                    """)
                    st.image("https://www.researchgate.net/profile/Chi-Shiang-Chan/publication/338733351/figure/fig1/AS:854050923429890@1580404329659/Visual-cryptography-scheme.png",
                           caption="Visual Cryptography Scheme (k-of-n)")

                if st.button("Next Step →", key="step2_next"):
                    st.session_state.current_step = 3
                    st.rerun()

        # Step 3: Decryption Process Visualization
        if st.session_state.current_step >= 3:
            st.markdown("<div style='text-align: center; font-size: 30px; margin: 20px 0'>⬇️</div>", 
                      unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown(f"### 3️⃣ Step {st.session_state.current_step}: Decryption Process")

                # Side-by-side image comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.image(st.session_state.input_image,
                           caption="Original Image",
                           use_column_width=True)
                with col2:
                    st.image(st.session_state.decrypted_image,
                           caption="Decrypted Image",
                           use_column_width=True)
                    similarity = structural_similarity(
                        np.array(st.session_state.input_image),
                        np.array(st.session_state.decrypted_image),
                        multichannel=True
                    )
                    st.success(f"✅ Decryption Successful (SSIM: {similarity:.4f})")

                # Key reconstruction demonstration
                with st.expander("🔑 Key Reconstruction Process"):
                    kcols = st.columns(3)
                    with kcols[0]:
                        st.image(st.session_state.encrypted_data['key_share_r'],
                               caption="Random Share (R)",
                               use_column_width=True)
                    with kcols[1]:
                        st.image(st.session_state.encrypted_data['key_share_p'],
                               caption="Secret Share (P)",
                               use_column_width=True)
                    with kcols[2]:
                        reconstructed_key = np.bitwise_xor(
                            st.session_state.encrypted_data['key_share_r'],
                            st.session_state.encrypted_data['key_share_p']
                        )
                        st.image(reconstructed_key,
                               caption="Reconstructed Key (R ⊕ P)",
                               use_column_width=True)

                # Full process reset
                if st.button("🔄 Start New Session"):
                    st.session_state.clear()
                    st.rerun()

if __name__ == "__main__":
    main()