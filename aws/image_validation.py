"""Standard-library header checks; decoding libraries load only on admitted work."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
MAX_BYTES = 8 * 1024 * 1024
MAX_SIDE = 8192


class ValidationError(ValueError):
    """An input fails the public API contract."""


def validate_image_header(data: bytes, max_bytes: int = MAX_BYTES,
                          max_side: int = MAX_SIDE) -> tuple[int, int, str]:
    if not isinstance(data, bytes) or not 0 < len(data) <= max_bytes:
        raise ValidationError("Image must contain nonempty JPEG or PNG data within its byte limit.")
    width = height = 0
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(data) < 33 or data[12:16] != b"IHDR" or struct.unpack(">I", data[8:12])[0] != 13:
            raise ValidationError("Invalid PNG header.")
        width, height = struct.unpack(">II", data[16:24])
        mime = "image/png"
    elif data.startswith(b"\xff\xd8"):
        mime = "image/jpeg"
        position = 2
        while position < len(data):
            if data[position] != 255:
                raise ValidationError("Invalid JPEG marker.")
            while position < len(data) and data[position] == 255:
                position += 1
            if position >= len(data):
                break
            marker = data[position]
            position += 1
            if marker in (0xD9, 0xDA):
                break
            if marker == 0x01 or 0xD0 <= marker <= 0xD7:
                continue
            if position + 2 > len(data):
                break
            length = int.from_bytes(data[position:position+2], "big")
            if length < 2 or position + length > len(data):
                raise ValidationError("Truncated JPEG segment.")
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                if length < 8:
                    raise ValidationError("Invalid JPEG dimensions.")
                height, width = struct.unpack(">HH", data[position+3:position+7])
                break
            position += length
    else:
        raise ValidationError("Only JPEG and PNG are accepted.")
    if not 1 <= width <= max_side or not 1 <= height <= max_side:
        raise ValidationError("Image dimensions must be 1 to 8192 pixels per side.")
    return width, height, mime


def validate_image_bytes(data: bytes, max_bytes: int = MAX_BYTES,
                         max_side: int = MAX_SIDE) -> np.ndarray:
    width, height, _ = validate_image_header(data, max_bytes, max_side)
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    image = cv2.imdecode(np.frombuffer(data, np.uint8),
                         cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None or image.shape[:2] != (height, width):
        raise ValidationError("Image cannot be decoded consistently with its header.")
    return image
