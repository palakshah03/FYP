import numpy as np
from PIL import Image
import base64
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import new as Random
from base64 import b64encode, b64decode
import cv2
import os

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
        
        # AES encryption
        aes_encrypted = AESCipher(img_base64, self.hashed_key).encrypt()
        
        # Visual cryptography
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

    def decrypt(self, encrypted_data):
        # Reconstruct visual key
        R = encrypted_data['key_share_r']
        P = encrypted_data['key_share_p']
        h, w = R.shape[:2]
        
        CK = np.ones((h, w, 1), dtype='uint8')
        for i in range(h):
            for j in range(w):
                CK[i][j][0] = P[i][j][0] ^ R[i][j][0]
        
        # Reconstruct key
        reconstructed_key = []
        for i in range(len(CK)):
            count = 0
            for j in range(len(CK[i])):
                if CK[i][j][0] == 0:
                    count += 1
            reconstructed_key.append(chr(count))
        
        reconstructed_key = "".join(reconstructed_key)
        
        # Decrypt visual shares
        shares = encrypted_data['visual_shares']
        (row, column, depth, share_size) = shares.shape
        shares_image = shares.copy()
        
        for i in range(share_size-1):
            shares_image[:,:,:,-1] = shares_image[:,:,:,-1] ^ shares_image[:,:,:,i]
        
        # Decrypt AES
        decrypted_base64 = AESCipher(encrypted_data['aes_cipher'], 
                                    hashlib.sha256(reconstructed_key.encode()).hexdigest()).decrypt()
        
        # Convert back to image
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
            share_size = int(input("\nEnter the number of shares to create (2-8): "))
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