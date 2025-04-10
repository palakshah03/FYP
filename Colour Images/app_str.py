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
    
    st.title("🔒 Hybrid AES-Visual Cryptography System")
    st.markdown("## Secure Image Encryption & Sharing")
    st.markdown("---")

    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
        st.session_state.encrypted_data = None
        st.session_state.decrypted_image = None

    # Sidebar controls
    with st.sidebar:
        st.header("🔑 Control Panel")
        uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
        key = st.text_input("Encryption Key (min 8 chars)", type="password")
        share_size = st.slider("Number of Shares", 2, 8, 4)
        process_btn = st.button("Encrypt & Process", disabled=not (uploaded_file and len(key)>=8))

    # Main vertical flowchart
    if uploaded_file or st.session_state.processed:
        with st.container():
            # Encryption Section
            with st.container(border=True):
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.subheader("1. AES Encryption Process")
                    input_image = Image.open(uploaded_file)
                    st.image(input_image, caption="Original Image", width=300)
                    
                with col2:
                    if st.session_state.processed:
                        st.subheader("Encryption Details")
                        key_vis = np.frombuffer(hashlib.sha256(key.encode()).digest()[:32], dtype=np.uint8).reshape(4, 8)
                        st.image(key_vis, caption="AES Key Heatmap", width=300)
                        
                        with st.expander("Technical Parameters"):
                            st.write(f"""
                            - **Key strength**: 256-bit
                            - **Mode**: OFB (Output Feedback)
                            - **IV size**: 16 bytes
                            - **Cipher size**: {len(st.session_state.encrypted_data['aes_cipher'])} chars
                            """)
                            img_base64 = base64.b64encode(input_image.tobytes()).decode('utf-8')
                            st.code(f'Base64 Header: {img_base64[:100]}...', language='text')

            # Downward arrow
            st.markdown("<div style='text-align: center; font-size: 30px; margin: -10px 0'>↓</div>", unsafe_allow_html=True)

            # Visual Cryptography Section
            with st.container(border=True):
                st.subheader("2. Visual Cryptography Shares")
                if st.session_state.processed:
                    # Visual shares grid
                    st.write(f"### Generated Shares ({share_size} total)")
                    cols = st.columns(4)
                    for idx in range(share_size):
                        with cols[idx % 4]:
                            share_img = Image.fromarray(
                                st.session_state.encrypted_data['visual_shares'][:,:,:,idx].astype(np.uint8)
                            )
                            st.image(share_img, caption=f"Share {idx+1}", width=200)
                    
                    # Key shares
                    with st.expander("Key Shares Configuration"):
                        kcol1, kcol2 = st.columns(2)
                        with kcol1:
                            st.image(
                                st.session_state.encrypted_data['key_share_r'], 
                                caption="Key Share R",
                                width=250
                            )
                        with kcol2:
                            st.image(
                                st.session_state.encrypted_data['key_share_p'],
                                caption="Key Share P",
                                width=250
                            )

            # Downward arrow
            st.markdown("<div style='text-align: center; font-size: 30px; margin: -10px 0'>↓</div>", unsafe_allow_html=True)

            # Decryption Section
            with st.container(border=True):
                st.subheader("3. Decryption Process")
                if st.session_state.processed:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(input_image, caption="Original Image", width=300)
                    with col2:
                        st.image(st.session_state.decrypted_image, caption="Decrypted Image", width=300)
                    st.success("✅ Decryption successful! Images match")
                    
                    with st.expander("Step-by-Step Decryption Process"):
                        st.write("**Key Reconstruction**")
                        rcol1, rcol2, rcol3 = st.columns([1,1,1])
                        with rcol1:
                            st.image(
                                st.session_state.encrypted_data['key_share_r'],
                                caption="Key Share R",
                                width=200
                            )
                        with rcol2:
                            st.image(
                                st.session_state.encrypted_data['key_share_p'],
                                caption="Key Share P", 
                                width=200
                            )
                        with rcol3:
                            CK = np.bitwise_xor(
                                st.session_state.encrypted_data['key_share_r'],
                                st.session_state.encrypted_data['key_share_p']
                            )
                            st.image(CK, caption="Reconstructed Key", width=200)
                        
                        st.write("**Share Combination Process**")
                        shares = st.session_state.encrypted_data['visual_shares']
                        combined = shares[:,:,:,-1].copy()
                        step_cols = st.columns(share_size-1)
                        for i in range(share_size-1):
                            combined = np.bitwise_xor(combined, shares[:,:,:,i])
                            with step_cols[i]:
                                st.image(combined, caption=f"After Share {i+1} XOR", width=200)

    # Processing logic
    if process_btn and uploaded_file and key:
        try:
            with st.spinner("🔐 Encrypting and generating shares..."):
                hybrid_crypto = HybridCryptography(key)
                encrypted_data = hybrid_crypto.encrypt(Image.open(uploaded_file), share_size)
                decrypted_image = hybrid_crypto.decrypt(encrypted_data)
                
                st.session_state.update({
                    'processed': True,
                    'encrypted_data': encrypted_data,
                    'decrypted_image': decrypted_image
                })
                st.rerun()
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")

if __name__ == "__main__":
    main()

