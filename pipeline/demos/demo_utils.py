import mimetypes
import sys
from typing import Union

import requests
from PIL import Image

requests.packages.urllib3.disable_warnings()


# --- Utility Functions ---
def print_colored(text, color_code):
    end_code = "\033[0m"  # Reset to default color
    print(f"{color_code}{text}{end_code}")

def get_content_type(file_path: str) -> str:
    return mimetypes.guess_type(file_path)[0]

def get_image(url: str) -> Union[Image.Image, list]:
    if not url.strip():  
        return Image.new("RGB", (224, 224))
        
    if "://" not in url:  # Local file
        content_type = get_content_type(url)
        if content_type is None:
            raise ValueError("Unable to determine content type of the local file.")
    else:  # Remote URL
        try:
            content_type = requests.head(url, stream=True).headers.get("Content-Type")
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch URL: {e}")
            
    if content_type is None:
        raise ValueError("Failed to fetch content type from remote URL.")
        
    if "image" in content_type:
        try:
            if "://" not in url:  # Local file
                return Image.open(url)
            else:  # Remote URL
                response = requests.get(url, stream=True)
                return Image.open(response.raw)
        except Exception as e:
            raise ValueError(f"Failed to open image: {e}")
    else:
        raise ValueError("Invalid content type. Expected image.")