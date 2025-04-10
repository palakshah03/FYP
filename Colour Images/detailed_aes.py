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

    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.update({
            'processed': False,
            'encrypted_data': None,
            'decrypted_image': None,
            'current_step': 1
        })

    # Sidebar controls with explanations
    with st.sidebar:
        st.header("🎛️ Control Panel")
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. Upload an image (PNG/JPG/JPEG)
            2. Set encryption key (min 8 characters)
            3. Choose number of visual shares
            4. Click 'Start Process' to begin
            """)

        uploaded_file = st.file_uploader("📤 Upload Image", type=['png', 'jpg', 'jpeg'])
        key = st.text_input("🔑 Encryption Key (min 8 chars)", type="password",
                          help="This key will be hashed using SHA-256 to generate AES-256 key")
        share_size = st.slider("🔢 Number of Visual Shares", 2, 8, 4,
                             help="Number of shares to split the encrypted image into")
        process_btn = st.button("🚀 Start Process", disabled=not (uploaded_file and len(key)>=8), key="start_process_btn") # Unique Key

    # Main vertical workflow
    if uploaded_file or st.session_state.processed:
        # Load image only once
        if uploaded_file and not st.session_state.processed:
            input_image = Image.open(uploaded_file).convert('RGB')
            st.session_state.input_image = input_image

        with st.container():
            # Step 1: AES Encryption
            with st.container(border=True):
                st.markdown(f"### 1️⃣ Step {st.session_state.current_step}: AES-256 Encryption")
                col1, col2 = st.columns([2, 3])

                with col1:
                    st.image(st.session_state.input_image,
                           caption="Original Image", width=300)

                with col2:
                    if st.session_state.processed:
                        # Improved AES Key Visualization
                        key_hash = hashlib.sha256(key.encode()).digest()
                        key_array = np.frombuffer(key_hash, dtype=np.uint8)

                        # Create figure with higher resolution
                        fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
                        im = ax.imshow(key_array.reshape(4, 8),
                                         cmap='coolwarm',
                                         interpolation='nearest')
                        plt.colorbar(im, ax=ax, label='Byte Value')
                        ax.set_title("AES-256 Key Heatmap (4x8)")
                        ax.axis('off')
                        st.pyplot(fig)


                        with st.expander("📚 Lecture: AES-256 Encryption"):
                            st.markdown("""
                            **AES-256 Encryption Process**
                            - Input Image → Byte Conversion → Base64 Encoding
                            - **Key Derivation**: SHA-256 hash of user-provided key
                            - **Parameters**:
                              - Mode: OFB (Output Feedback)
                              - IV: 16-byte random initialization vector
                              - Block Size: 128 bits
                              - Key Schedule: 14 rounds of transformation
                            - Mathematical Foundation:
                              ```
                              C_i = P_i ⊕ O_i
                              O_i = Encrypt_K(O_{i-1})
                              ```
                            Where:
                            - \( C_i \) = Cipher block
                            - \( P_i \) = Plaintext block
                            - \( O_i \) = Output feedback
                            """)

                        st.button("Next Step →", on_click=lambda: st.session_state.update(current_step=2), key="next_step_1") # Unique Key

            # Step 2: Visual Cryptography
            if st.session_state.current_step >= 2:
                st.markdown("<div style='text-align: center; font-size: 30px; margin: 10px 0'>↓</div>",
                          unsafe_allow_html=True)

                with st.container(border=True):
                    st.markdown(f"### 2️⃣ Step {st.session_state.current_step}: Visual Cryptography Shares")

                    if st.session_state.processed:
                        # Visual Shares Gallery
                        st.markdown("#### 🖼️ Generated Visual Shares")
                        cols = st.columns(4)
                        for idx in range(share_size):
                            with cols[idx % 4]:
                                if 'visual_shares' in st.session_state.encrypted_data:
                                    share_img = Image.fromarray(
                                        st.session_state.encrypted_data['visual_shares'][:,:,:,idx].astype(np.uint8)
                                    )
                                    st.image(share_img, caption=f"Share {idx+1}", use_column_width=True)

                                    # Interactive share inspection
                                    with st.expander(f"🔍 Inspect Share {idx+1}"):
                                        st.image(share_img, use_column_width=True)
                                        st.markdown(f"""
                                        - Dimensions: {share_img.size[0]}x{share_img.size[1]}
                                        - Color Mode: RGB
                                        - Unique Patterns: {np.unique(share_img).size}
                                        """)
                                else:
                                    st.warning("Visual shares not found. Please process the image first.")


                        with st.expander("📚 Lecture: Visual Cryptography"):
                            st.markdown("""
                            **Visual Cryptography Scheme**
                            - Split encrypted data into multiple shares
                            - **XOR-based Reconstruction**:
                              ```
                              Original = Share_1 ⊕ Share_2 ⊕ ... ⊕ Share_n
                              ```
                            - Key Shares:
                              - R: Random pattern matrix
                              - P: Secret pattern matrix
                              - Combined: \( R ⊕ P = Secret \)
                            """)

                        st.button("Next Step →", on_click=lambda: st.session_state.update(current_step=3), key="next_step_2") # Unique Key

            # Step 3: Decryption Process
            if st.session_state.current_step >= 3:
                st.markdown("<div style='text-align: center; font-size: 30px; margin: 10px 0'>↓</div>",
                          unsafe_allow_html=True)

                with st.container(border=True):
                    st.markdown(f"### 3️⃣ Step {st.session_state.current_step}: Decryption Process")

                    if st.session_state.processed:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(st.session_state.input_image,
                                   caption="Original Image", use_column_width=True)
                        with col2:
                            st.image(st.session_state.decrypted_image,
                                   caption="Decrypted Image", use_column_width=True)
                            st.success("✅ Decryption Successful")

                        with st.expander("🔬 Step-by-Step Decryption Demo"):
                            st.markdown("#### Key Reconstruction")
                            kcol1, kcol2, kcol3 = st.columns(3)
                            with kcol1:
                                st.image(st.session_state.encrypted_data['key_share_r'],
                                       caption="Key Share R (Random)")
                            with kcol2:
                                st.image(st.session_state.encrypted_data['key_share_p'],
                                       caption="Key Share P (Secret)")
                            with kcol3:
                                reconstructed_key = np.bitwise_xor(
                                    st.session_state.encrypted_data['key_share_r'],
                                    st.session_state.encrypted_data['key_share_p']
                                )
                                st.image(reconstructed_key,
                                       caption="Reconstructed Key (R ⊕ P)")

                            st.markdown("#### Share Combination Process")
                            shares = st.session_state.encrypted_data['visual_shares']
                            combined = shares[:,:,:,-1].copy()
                            step_cols = st.columns(share_size)
                            for i in range(share_size-1):
                                combined = np.bitwise_xor(combined, shares[:,:,:,i])
                                with step_cols[i]:
                                    st.image(combined,
                                           caption=f"After Share {i+1} XOR",
                                           use_column_width=True)

                        with st.expander("📚 Lecture: Decryption Process"):
                            st.markdown("""
                            **Decryption Workflow**
                            1. Combine required number of shares using XOR
                            2. Reconstruct AES key using key shares
                            3. AES decryption with OFB mode:
                               ```
                               P_i = C_i ⊕ O_i
                               O_i = Encrypt_K(O_{i-1})
                               ```
                            4. Base64 decoding → Image reconstruction
                            """)

    # Processing logic
    if process_btn and uploaded_file and key:
        try:
            with st.spinner("🔨 Processing - Generating AES key and visual shares..."):
                hybrid_crypto = HybridCryptography(key)  # Ensure HybridCryptography is correctly implemented and imported
                encrypted_data = hybrid_crypto.encrypt(st.session_state.input_image, share_size)
                decrypted_image = hybrid_crypto.decrypt(encrypted_data)

                st.session_state.update({
                    'processed': True,
                    'encrypted_data': encrypted_data,
                    'decrypted_image': decrypted_image,
                    'current_step': 1
                })
                #st.session_state.processed = True #Setting this to true lets me test the website
                st.rerun()

        except Exception as e:
            st.error(f"🚨 Processing error: {str(e)}")

if __name__ == "__main__":
    main()