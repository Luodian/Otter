import base64
from io import BytesIO
from PIL import Image


def resize_image(image, target_size=(224, 224)):
    """
    Resizes the given image to the target size using the Lanczos algorithm.

    Args:
        image (PIL.Image.Image): The input image to be resized.
        target_size (tuple[int, int]): The target size to which the image should be resized.
            Defaults to (224, 224).

    Returns:
        PIL.Image.Image: The resized image.

    """
    if image.size != target_size:
        return image.resize(target_size, Image.LANCZOS)
    return image


def process_image(img: Image):
    """
    Processes the input image by resizing it, converting it to RGB mode, and encoding it as base64.

    Args:
        img (PIL.Image.Image): The input image to be processed.

    Returns:
        str: The base64 encoded string representation of the processed image.

    """
    resized_img = resize_image(img)
    if resized_img.mode != "RGB":
        resized_img = resized_img.convert("RGB")
    buffer = BytesIO()
    resized_img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_base64
