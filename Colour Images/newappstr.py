import numpy as np
from PIL import Image
import base64
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import new as Random
from base64 import b64encode, b64decode
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class AESCipher:
    def __init__(self, data, key):
        self.block_size = 16
        self.data = data
        self.key = hashlib.sha256(key.encode()).digest()[:32]
        self.pad = lambda s: s + (self.block_size - len(s) % self.block_size) * chr(self.block_size - len(s) % self.block_size)
        self.unpad = lambda s: s[:-ord(s[len(s) - 1:])]

    def visualize_encryption(self, original_image):
        """Visualize the AES encryption process"""
        plt.figure(figsize=(15, 8))
        gs = GridSpec(2, 3, figure=plt.gcf())
        
        # Original Image
        plt.subplot(gs[0, 0])
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Show key visualization
        plt.subplot(gs[0, 1])
        key_vis = np.frombuffer(self.key, dtype=np.uint8).reshape(4, 8)
        plt.imshow(key_vis, cmap='viridis')
        plt.title('AES Key Visualization\n(32 bytes)')
        plt.colorbar()
        plt.axis('off')
        
        # Convert to base64 and show sample
        img_base64 = base64.b64encode(original_image.tobytes()).decode('utf-8')
        plt.subplot(gs[0, 2])
        plt.text(0.1, 0.5, f'Base64 Sample:\n{img_base64[:100]}...', 
                fontsize=8, wrap=True)
        plt.axis('off')
        plt.title('Base64 Conversion')
        
        # Encrypt
        plain_text = self.pad(img_base64)
        iv = Random().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_OFB, iv)
        encrypted = b64encode(iv + cipher.encrypt(plain_text.encode())).decode()
        
        # Show encryption process steps
        plt.subplot(gs[1, :])
        steps_text = (
            f"Encryption Steps:\n\n"
            f"1. Image Size: {original_image.size}\n"
            f"2. Base64 Length: {len(img_base64)} chars\n"
            f"3. Padded Length: {len(plain_text)} chars\n"
            f"4. IV Size: {len(iv)} bytes\n"
            f"5. Final Encrypted Length: {len(encrypted)} chars\n\n"
            f"Sample of encrypted data:\n{encrypted[:100]}..."
        )
        plt.text(0.1, 0.1, steps_text, fontsize=10, wrap=True)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return encrypted

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
        
        # Generate key pattern
        for i in range(h):
            j = ord(self.key[i])
            for k in range(width):
                if k < j:
                    C[i][k][0] = 0
                else:
                    break
                    
        # Generate random share R and computed share P
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
        
        # Create AES cipher and visualize encryption
        aes_cipher = AESCipher(img_base64, self.hashed_key)
        aes_encrypted = aes_cipher.visualize_encryption(input_image)
        
        # Continue with visual cryptography
        image_array = np.array(input_image)
        (row, column, depth) = image_array.shape
        shares = np.random.randint(0, 256, size=(row, column, depth, share_size))
        shares[:,:,:,-1] = image_array.copy()
        
        for i in range(share_size-1):
            shares[:,:,:,-1] = shares[:,:,:,-1] ^ shares[:,:,:,i]
        
        # Generate visual key shares
        R, P = self.generate_visual_key(row, 255)
        
        return {
            'aes_cipher': aes_encrypted,
            'visual_shares': shares,
            'key_share_r': R,
            'key_share_p': P
        }

    def visualize_reconstruction(self, encrypted_data):
        """Visualize the decryption and reconstruction process for four shares"""
        shares = encrypted_data['visual_shares']
        R = encrypted_data['key_share_r']
        P = encrypted_data['key_share_p']
        
        # Create a larger figure with a 3x3 grid
        plt.figure(figsize=(15, 15))
        gs = GridSpec(3, 3, figure=plt.gcf())
        
        # First row: Show first three random shares
        for i in range(3):
            plt.subplot(gs[0, i])
            plt.imshow(shares[:,:,:,i])
            plt.title(f'Share {i + 1} (Random)')
            plt.axis('off')
        
        # Second row: Show combined share, key shares
        plt.subplot(gs[1, 0])
        plt.imshow(shares[:,:,:,-1])
        plt.title('Share 4 (Combined)')
        plt.axis('off')
        
        plt.subplot(gs[1, 1])
        plt.imshow(R)
        plt.title('Key Share R')
        plt.axis('off')
        
        plt.subplot(gs[1, 2])
        plt.imshow(P)
        plt.title('Key Share P')
        plt.axis('off')
        
        # Third row: Show reconstruction steps and final image
        shares_image = shares.copy()
        for i in range(shares.shape[3]-1):
            shares_image[:,:,:,-1] = shares_image[:,:,:,-1] ^ shares_image[:,:,:,i]
        
        plt.subplot(gs[2, 0])
        plt.imshow(shares_image[:,:,:,-1])
        plt.title('After XOR Operation')
        plt.axis('off')
        
        # Show key reconstruction
        h, w = R.shape[:2]
        CK = np.ones((h, w, 1), dtype='uint8')
        for i in range(h):
            for j in range(w):
                CK[i][j][0] = P[i][j][0] ^ R[i][j][0]
        
        plt.subplot(gs[2, 1])
        plt.imshow(CK)
        plt.title('Reconstructed Key Pattern')
        plt.axis('off')
        
        # Get final decrypted image
        reconstructed_key = []
        for i in range(len(CK)):
            count = 0
            for j in range(len(CK[i])):
                if CK[i][j][0] == 0:
                    count += 1
            reconstructed_key.append(chr(count))
        
        reconstructed_key = "".join(reconstructed_key)
        decrypted_base64 = AESCipher(encrypted_data['aes_cipher'], reconstructed_key).decrypt()
        decrypted_image = base64.b64decode(decrypted_base64)
        plt.subplot(gs[2, 2])
        plt.imshow(decrypted_image)
        plt.title('Final Decrypted Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
