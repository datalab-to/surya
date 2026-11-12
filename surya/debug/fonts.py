from typing import List, Optional
import os
import tempfile
import httpx

from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()


def get_font_path(langs: Optional[List[str]] = None) -> Optional[str]:
    font_path = settings.RECOGNITION_RENDER_FONTS["all"]
    if langs is not None:
        for k in settings.RECOGNITION_RENDER_FONTS:
            if k in langs and len(langs) == 1:
                font_path = settings.RECOGNITION_RENDER_FONTS[k]
                break

    if not os.path.exists(font_path):
        font_dir = os.path.dirname(font_path)
        os.makedirs(font_dir, exist_ok=True)
        font_dl_path = (
            f"{settings.RECOGNITION_FONT_DL_BASE}/{os.path.basename(font_path)}"
        )
        fd, tmp_path = tempfile.mkstemp(dir=font_dir, suffix=".tmp")
        try:
            with (
                os.fdopen(fd, "wb") as f,
                httpx.stream(
                    "GET", font_dl_path, follow_redirects=True, timeout=60.0
                ) as r,
            ):
                r.raise_for_status()
                for chunk in r.iter_bytes(chunk_size=8192):
                    f.write(chunk)
            os.replace(tmp_path, font_path)
        except Exception as e:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            logger.warning(f"Could not download font from {font_dl_path}: {e}")
            return None

    return font_path
