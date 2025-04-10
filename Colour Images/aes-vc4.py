import numpy as np
from PIL import Image
import base64
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import new as Random
from base64 import b64encode, b64decode
import cv2
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
        print("\nVisualizing AES Encryption Process...")
        aes_encrypted = aes_cipher.visualize_encryption(input_image)
        
        # Continue with visual cryptography
        image_array = np.array(input_image)
        (row, column, depth) = image_array.shape
        
        # Generate initial shares as pure noise
        shares = np.random.randint(0, 256, size=(row, column, depth, share_size))
        
        # Create distorted grayscale version for combined share
        # Convert to grayscale first
        grayscale = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        grayscale = np.stack([grayscale] * 3, axis=-1)  # Convert back to 3 channels
        
        # Add various distortion effects
        distorted = grayscale.astype(np.float32)  # Convert to float32 for calculations
        
        # 1. Add salt and pepper noise
        salt_pepper_prob = 0.2
        salt_mask = np.random.random(distorted.shape[:2]) < salt_pepper_prob/2
        pepper_mask = np.random.random(distorted.shape[:2]) < salt_pepper_prob/2
        
        for i in range(3):  # Apply to each channel
            distorted_channel = distorted[:,:,i]
            distorted_channel[salt_mask] = 255
            distorted_channel[pepper_mask] = 0
        
        # 2. Add Gaussian noise
        gaussian_noise = np.random.normal(0, 25, distorted.shape)
        distorted = np.clip(distorted + gaussian_noise, 0, 255)
        
        # 3. Add block distortion
        block_size = 8
        for i in range(0, row, block_size):
            for j in range(0, column, block_size):
                if np.random.random() > 0.7:  # 30% chance to distort each block
                    for c in range(3):  # Process each channel separately
                        block = distorted[i:i+block_size, j:j+block_size, c]
                        if block.size > 0:
                            effect = np.random.choice(['invert', 'blur', 'shift'])
                            
                            if effect == 'invert':
                                distorted[i:i+block_size, j:j+block_size, c] = 255 - block
                            elif effect == 'blur':
                                if block.shape[0] > 1 and block.shape[1] > 1:
                                    blurred = cv2.blur(block.astype(np.uint8).reshape(block.shape), (3,3))
                                    distorted[i:i+block_size, j:j+block_size, c] = blurred
                            elif effect == 'shift':
                                shift = np.random.randint(-30, 30)
                                distorted[i:i+block_size, j:j+block_size, c] = np.clip(block + shift, 0, 255)
        
        # 4. Add scanning lines effect
        for i in range(0, row, 4):  # Every 4th row
            if i + 2 < row:  # Make sure we don't go out of bounds
                distorted[i:i+2, :] = np.clip(distorted[i:i+2, :] * 0.7, 0, 255)
        
        # 5. Add vertical glitch effect
        num_glitches = 20
        for _ in range(num_glitches):
            x = np.random.randint(0, column-10)
            width = np.random.randint(5, 20)
            shift = np.random.randint(5, 15)  # Reduced shift range
            if x + width < column:
                # Shift up or down randomly
                direction = np.random.choice([-1, 1])
                slice_data = distorted[:, x:x+width].copy()
                if direction == 1 and x + shift + width < column:
                    distorted[:, x+shift:x+shift+width] = slice_data
                elif direction == -1 and x - shift >= 0:
                    distorted[:, x-shift:x-shift+width] = slice_data
        
        # Final clipping and conversion to uint8
        distorted = np.clip(distorted, 0, 255).astype(np.uint8)
        
        # Store the distorted grayscale image as the combined share
        shares[:,:,:,-1] = distorted
        
        # Generate visual key shares
        R, P = self.generate_visual_key(row, 255)
        
        return {
            'aes_cipher': aes_encrypted,
            'visual_shares': shares,
            'key_share_r': R,
            'key_share_p': P
        }

    def visualize_reconstruction(self, encrypted_data):
        """Visualize the decryption and reconstruction process for shares"""
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
        # Show reconstruction process with modified operations
        combined_share = shares[:,:,:,-1].copy()
        plt.subplot(gs[2, 0])
        plt.imshow(combined_share)
        plt.title('After Combined Operations')
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
        
        # Get final decrypted image using AES
        reconstructed_key = []
        for i in range(len(CK)):
            count = 0
            for j in range(len(CK[i])):
                if CK[i][j][0] == 0:
                    count += 1
            reconstructed_key.append(chr(count))
        
        reconstructed_key = "".join(reconstructed_key)
        decrypted_base64 = AESCipher(encrypted_data['aes_cipher'], 
                                    hashlib.sha256(reconstructed_key.encode()).hexdigest()).decrypt()
        
        img_bytes = base64.b64decode(decrypted_base64.encode('utf-8'))
        final_image = Image.frombytes('RGB', (shares.shape[1], shares.shape[0]), img_bytes)
        
        plt.subplot(gs[2, 2])
        plt.imshow(final_image)
        plt.title('Final Decrypted Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return final_image
            
    def decrypt(self, encrypted_data, visualize=True):
        """
        Decrypt the image with optional visualization
        Args:
            encrypted_data: Dictionary containing encrypted data
            visualize: Boolean to control visualization
        """
        if visualize:
            return self.visualize_reconstruction(encrypted_data)
        
        # Original decryption code
        R = encrypted_data['key_share_r']
        P = encrypted_data['key_share_p']
        h, w = R.shape[:2]
        
        CK = np.ones((h, w, 1), dtype='uint8')
        for i in range(h):
            for j in range(w):
                CK[i][j][0] = P[i][j][0] ^ R[i][j][0]
        
        reconstructed_key = []
        for i in range(len(CK)):
            count = 0
            for j in range(len(CK[i])):
                if CK[i][j][0] == 0:
                    count += 1
            reconstructed_key.append(chr(count))
        
        reconstructed_key = "".join(reconstructed_key)
        
        # Decrypt using AES
        decrypted_base64 = AESCipher(encrypted_data['aes_cipher'], 
                                    hashlib.sha256(reconstructed_key.encode()).hexdigest()).decrypt()
        
        img_bytes = base64.b64decode(decrypted_base64.encode('utf-8'))
        final_image = Image.frombytes('RGB', (column, row), img_bytes)
        
        return final_image

def try_image_path(base_path):
    """Try common image extensions if no extension is provided"""
    common_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    
    # If path already has an extension, try it first
    if os.path.splitext(base_path)[1].lower() in common_extensions:
        if os.path.exists(base_path):
            return base_path
        return None
    
    # Try adding common extensions
    for ext in common_extensions:
        test_path = base_path + ext
        if os.path.exists(test_path):
            return test_path
    
    return None

def get_user_input():
    """Get user inputs with proper validation"""
    while True:
        # Get image path
        image_path = input("\nEnter the path to your image file: ").strip()
        
        # Remove quotes if present
        image_path = image_path.strip('"\'')
        
        # Try to find the image
        valid_path = try_image_path(image_path)
        
        if valid_path:
            print(f"Found image at: {valid_path}")
            break
        
        print("\nError: Image file not found. Please ensure:")
        print("1. The file path is correct")
        print("2. The file has a valid image extension (.png, .jpg, .jpeg, .bmp, .gif)")
        print("3. You have permission to access the file")
        print("\nTip: You can copy the full path from File Explorer and paste it here")

    while True:
        try:
            # Get number of shares
            share_size = int(input("\nEnter the number of shares to create (3-8): "))
            if 2 <= share_size <= 8:
                break
            print("Error: Number of shares must be between 2 and 8.")
        except ValueError:
            print("Error: Please enter a valid number.")

    while True:
        # Get encryption key
        key = input("\nEnter your encryption key (minimum 8 characters): ").strip()
        if len(key) >= 8:
            break
        print("Error: Key must be at least 8 characters long.")

    return valid_path, share_size, key

def create_output_directory():
    """Create output directory if it doesn't exist"""
    output_dir = "encrypted_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def main():
    print("=" * 50)
    print("Hybrid AES-Visual Cryptography System")
    print("=" * 50)
    
    try:
        # Get user inputs
        image_path, share_size, key = get_user_input()
        
        # Create output directory
        output_dir = create_output_directory()
        
        # Load and process image
        print("\nProcessing image...")
        try:
            input_image = Image.open(image_path)
            # Convert to RGB if needed
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
            print(f"Image size: {input_image.size}")
        except Exception as e:
            print(f"\nError loading image: {str(e)}")
            print("Please ensure the file is a valid image file.")
            return
        
        # Initialize cryptography system
        hybrid_crypto = HybridCryptography(key)
        
        # Encrypt
        print("\nEncrypting...")
        encrypted_data = hybrid_crypto.encrypt(input_image, share_size)
        
        # Save shares
        shares = encrypted_data['visual_shares']
        print(f"\nGenerating {share_size} visual shares...")
        for ind in range(shares.shape[3]):
            image = Image.fromarray(shares[:,:,:,ind].astype(np.uint8))
            name = f"{output_dir}/Share_{ind+1}.png"
            image.save(name)
            print(f"Saved share {ind+1} to {name}")
        
        # Save key shares
        print("\nSaving key shares...")
        cv2.imwrite(f'{output_dir}/Key_Share_R.png', encrypted_data['key_share_r'])
        cv2.imwrite(f'{output_dir}/Key_Share_P.png', encrypted_data['key_share_p'])
        
        # Save AES cipher
        print("Saving AES cipher...")
        with open(f'{output_dir}/aes_cipher.txt', 'w') as f:
            f.write(encrypted_data['aes_cipher'])
        
        # Decrypt
        print("\nTesting decryption...")
        decrypted_image = hybrid_crypto.decrypt(encrypted_data)
        decrypted_image.save(f'{output_dir}/Decrypted_Image.png')
        
        print("\nProcess completed successfully!")
        print(f"\nAll files have been saved to the '{output_dir}' directory:")
        print(f"- {share_size} visual shares (Share_1.png to Share_{share_size}.png)")
        print("- 2 key shares (Key_Share_R.png and Key_Share_P.png)")
        print("- AES cipher text (aes_cipher.txt)")
        print("- Decrypted image (Decrypted_Image.png)")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()

#C:\Users\palak\OneDrive\Pictures\wedding\sangeet
# C:\Users\palak\OneDrive\Pictures\Screenshots\Screenshots\beautiful_flower
# C:\Users\palak\OneDrive\Pictures\Screenshots\Screenshots\cartoon_running.gif

#this was my computing classes give streamlit output from def main acc to this only and ensure everything is outputted perfectly