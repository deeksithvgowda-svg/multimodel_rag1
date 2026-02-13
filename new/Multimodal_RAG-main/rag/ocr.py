import io

import numpy as np
from PIL import Image


def extract_text_from_image(image_bytes):
    try:
        import easyocr
    except ImportError:
        return ""

    reader = easyocr.Reader(["en"])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image)
    result = reader.readtext(image_array)

    text = " ".join([t for (_, t, _) in result])
    return text.strip()
