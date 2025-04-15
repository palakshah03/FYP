import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import base64

def generate_aes_key():
    """Generate a random 128-bit AES key"""
    return get_random_bytes(16)  # 16 bytes = 128 bits

def encrypt_image_aes(image_path, key):
    """Encrypt an image using AES-128 in CBC mode"""
    try:
        # Read image as binary data
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        if len(image_data) == 0:
            raise ValueError("Image file is empty")
        
        # Generate random IV
        iv = get_random_bytes(AES.block_size)
        
        # Create cipher object and encrypt
        cipher = AES.new(key, AES.MODE_CBC, iv)
        encrypted_data = cipher.encrypt(pad(image_data, AES.block_size))
        
        # Return IV + encrypted data
        return iv + encrypted_data
    except Exception as e:
        raise Exception(f"AES encryption failed: {str(e)}")

def decrypt_image_aes(encrypted_data, key, output_path):
    """Decrypt AES-128 encrypted image data"""
    try:
        # Verify input data
        if len(encrypted_data) < AES.block_size:
            raise ValueError(f"Encrypted data too short ({len(encrypted_data)} bytes) to contain IV")
        
        # Extract IV and encrypted data
        iv = encrypted_data[:AES.block_size]
        ciphertext = encrypted_data[AES.block_size:]
        
        if len(ciphertext) % AES.block_size != 0:
            raise ValueError(f"Ciphertext length {len(ciphertext)} is not a multiple of block size")
        
        # Create cipher object and decrypt
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_data = unpad(cipher.decrypt(ciphertext), AES.block_size)
        
        # Verify decrypted data looks like an image
        if len(decrypted_data) < 8:
            raise ValueError("Decrypted data too small to be valid")
        
        # Check for common image headers
        if not (decrypted_data.startswith(b'\x89PNG') or   # PNG
           not decrypted_data.startswith(b'\xff\xd8') or   # JPEG
           not decrypted_data.startswith(b'BM')):          # BMP
            raise ValueError("Decrypted data doesn't contain valid image header")
        
        # Save decrypted image
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        return True
    except ValueError as e:
        raise ValueError(f"Decryption validation failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Decryption process failed: {str(e)}")

def key_to_base64(key):
    """Convert bytes key to base64 string for storage"""
    return base64.b64encode(key).decode('utf-8')

def base64_to_key(key_str):
    """Convert base64 string back to bytes key"""
    try:
        return base64.b64decode(key_str.encode('utf-8'))
    except Exception as e:
        raise ValueError(f"Invalid base64 key: {str(e)}")