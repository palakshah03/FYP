import cv2
import numpy as np
import os
import shutil
from typing import List, Tuple

def binary_to_image(binary_data: bytes) -> np.ndarray:
    """Convert binary data to an image format with length information"""
    # Store original length in first 4 bytes (big-endian)
    length = len(binary_data)
    length_bytes = length.to_bytes(4, byteorder='big')
    combined_data = length_bytes + binary_data
    
    # Calculate required padding
    total_bytes = len(combined_data)
    bytes_per_pixel = 3  # Using RGB format
    pixels_needed = (total_bytes + bytes_per_pixel - 1) // bytes_per_pixel
    
    # Create square image that can fit all data
    side_length = int(np.ceil(np.sqrt(pixels_needed)))
    total_pixels = side_length * side_length
    padding_needed = total_pixels * bytes_per_pixel - total_bytes
    
    # Add padding and reshape to image
    padded_data = combined_data + bytes(padding_needed)
    img = np.frombuffer(padded_data, dtype=np.uint8).reshape((side_length, side_length, bytes_per_pixel))
    
    return img

def image_to_binary(img: np.ndarray) -> bytes:
    """Convert image back to binary data"""
    # Get all image data as bytes
    all_data = img.tobytes()
    
    # First 4 bytes contain original length
    original_length = int.from_bytes(all_data[:4], byteorder='big')
    
    # Return original data without padding
    return all_data[4:4+original_length]

def generate_shares(data_path: str, n: int, output_dir: str) -> Tuple[List[str], str]:
    """
    Generate n visual cryptography shares from binary data
    Returns (paths to shares, folder containing all shares)
    """
    # Read the binary data
    with open(data_path, 'rb') as f:
        binary_data = f.read()
    
    # Convert to image format
    img = binary_to_image(binary_data)
    
    # Create output directory for shares
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    shares_dir = os.path.join(output_dir, f"{base_name}_shares")
    os.makedirs(shares_dir, exist_ok=True)
    
    share_paths = []
    
    # Generate n-1 random shares
    shares = []
    for i in range(n-1):
        share = np.random.randint(0, 256, size=img.shape, dtype=np.uint8)
        shares.append(share)
        share_path = os.path.join(shares_dir, f"share_{i+1}.png")
        cv2.imwrite(share_path, share)
        share_paths.append(share_path)
    
    # Generate the last share by XORing all previous shares with the image
    last_share = img.copy()
    for share in shares:
        last_share = cv2.bitwise_xor(last_share, share)
    
    share_path = os.path.join(shares_dir, f"share_{n}.png")
    cv2.imwrite(share_path, last_share)
    share_paths.append(share_path)
    
    return share_paths, shares_dir

def combine_shares(share_paths: List[str]) -> Tuple[bool, bytes, str]:
    """
    Combine visual cryptography shares to reconstruct the binary data
    Returns (success, combined_data, error_message)
    """
    if len(share_paths) < 2:
        return False, b'', "At least 2 shares are required"
    
    try:
        # Read all shares
        shares = []
        for i, path in enumerate(share_paths):
            share = cv2.imread(path, cv2.IMREAD_COLOR)
            if share is None:
                return False, b'', f"Could not read share {i+1} (invalid image file)"
            shares.append(share)
        
        # Check all shares have same dimensions
        base_shape = shares[0].shape
        for i, share in enumerate(shares[1:]):
            if share.shape != base_shape:
                return False, b'', f"Share {i+2} has different dimensions ({share.shape}) than share 1 ({base_shape})"
        
        # Combine shares
        combined = shares[0].copy()
        for share in shares[1:]:
            combined = cv2.bitwise_xor(combined, share)
        
        # Convert back to binary
        try:
            binary_data = image_to_binary(combined)
            if len(binary_data) < 10:  # Minimum reasonable size
                return False, b'', "Reconstructed data too small to be valid"
            return True, binary_data, ""
        except Exception as e:
            return False, b'', f"Data reconstruction failed: {str(e)}"
            
    except Exception as e:
        return False, b'', f"Unexpected error during combination: {str(e)}"