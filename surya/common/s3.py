import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()

# Lock file expiration time in seconds (10 minutes)
LOCK_EXPIRATION = 600


def join_urls(url1: str, url2: str):
    url1 = url1.rstrip("/")
    url2 = url2.lstrip("/")
    return f"{url1}/{url2}"


def get_model_name(pretrained_model_name_or_path: str):
    return pretrained_model_name_or_path.split("/")[0]


def download_file(remote_path: str, local_path: str, chunk_size: int = 1024 * 1024):
    local_path = Path(local_path)
    try:
        response = requests.get(remote_path, stream=True, allow_redirects=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Get file size from headers for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        # Create progress bar with file name and size info
        filename = local_path.name
        pbar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {filename}",
            miniters=1
        )

        with open(local_path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    pbar.update(len(chunk))
        
        pbar.close()
        return local_path
    except Exception as e:
        if local_path.exists():
            local_path.unlink()
        logger.error(f"Download error for file {remote_path}: {str(e)}")
        raise


def check_manifest(local_dir: str):
    local_dir = Path(local_dir)
    manifest_path = local_dir / "manifest.json"
    if not os.path.exists(manifest_path):
        return False

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        for file in manifest["files"]:
            if not os.path.exists(local_dir / file):
                return False
    except Exception:
        return False

    return True


def download_directory(remote_path: str, local_dir: str):
    model_name = get_model_name(remote_path)
    s3_url = join_urls(settings.S3_BASE_URL, remote_path)
    # Check to see if it's already downloaded
    model_exists = check_manifest(local_dir)
    if model_exists:
        return

    # Use tempfile.TemporaryDirectory to automatically clean up
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the manifest file
        manifest_file = join_urls(s3_url, "manifest.json")
        manifest_path = os.path.join(temp_dir, "manifest.json")
        download_file(manifest_file, manifest_path)

        # List and download all files
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        pbar = tqdm(
            desc=f"Downloading {model_name} model to {local_dir}",
            total=len(manifest["files"]),
        )

        with ThreadPoolExecutor(
            max_workers=settings.PARALLEL_DOWNLOAD_WORKERS
        ) as executor:
            futures = []
            for file in manifest["files"]:
                remote_file = join_urls(s3_url, file)
                local_file = os.path.join(temp_dir, file)
                futures.append(executor.submit(download_file, remote_file, local_file))

            for future in futures:
                future.result()
                pbar.update(1)

        pbar.close()

        # Move all files to new directory
        for file in os.listdir(temp_dir):
            shutil.move(os.path.join(temp_dir, file), local_dir)


class S3DownloaderMixin:
    s3_prefix = "s3://"

    @classmethod
    def get_local_path(cls, pretrained_model_name_or_path) -> str:
        if pretrained_model_name_or_path.startswith(cls.s3_prefix):
            pretrained_model_name_or_path = pretrained_model_name_or_path.replace(
                cls.s3_prefix, ""
            )
            cache_dir = settings.MODEL_CACHE_DIR
            local_path = os.path.join(cache_dir, pretrained_model_name_or_path)
            os.makedirs(local_path, exist_ok=True)
        else:
            local_path = ""
        return local_path

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Allow loading models directly from the hub, or using s3
        if not pretrained_model_name_or_path.startswith(cls.s3_prefix):
            return super().from_pretrained(
                pretrained_model_name_or_path, *args, **kwargs
            )

        local_path = cls.get_local_path(pretrained_model_name_or_path)
        pretrained_model_name_or_path = pretrained_model_name_or_path.replace(
            cls.s3_prefix, ""
        )

        # Retry logic for downloading the model folder
        retries = 3
        delay = 5
        attempt = 0
        success = False
        while not success and attempt < retries:
            try:
                download_directory(pretrained_model_name_or_path, local_path)
                success = True  # If download succeeded
            except Exception as e:
                logger.error(
                    f"Error downloading model from {pretrained_model_name_or_path}. Attempt {attempt + 1} of {retries}. Error: {e}"
                )
                attempt += 1
                if attempt < retries:
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)  # Wait before retrying
                else:
                    logger.error(
                        f"Failed to download {pretrained_model_name_or_path} after {retries} attempts."
                    )
                    raise e  # Reraise exception after max retries

        result = super().from_pretrained(local_path, *args, **kwargs)
        cls._materialize_weights_if_needed(result, local_path)
        return result

    @staticmethod
    def _materialize_weights_if_needed(result, local_path: str) -> None:
        """Repair transformers >= 5.0's silent weight-load failure on
        surya's custom (vendored) model classes.

        transformers 5.x's mandatory meta-device from_pretrained path
        silently fails to materialize checkpoint weights into these
        models -- it reports a clean load (no missing/unexpected keys,
        no error) while leaving every parameter at its random init,
        which collapses model outputs (e.g. the OCR-error DistilBert and
        the EfficientViT text detector). Loading the local single-file
        safetensors checkpoint directly materializes them correctly
        (verified bit-exact).

        Scoped narrowly so nothing else changes:
          - transformers < 5: no-op (4.x already loads weights eagerly),
            so 4.x behavior is byte-for-byte unchanged.
          - non-nn.Module results: no-op, so the config / tokenizer /
            processor classes that also use this mixin are untouched.
          - sharded or non-safetensors checkpoints: no-op (no local
            model.safetensors to reload from).
        Best-effort: never raises; a failed repair just logs.
        """
        try:
            import transformers

            if int(transformers.__version__.split(".")[0]) < 5:
                return
            import torch

            if not isinstance(result, torch.nn.Module):
                return
            weights_file = os.path.join(local_path, "model.safetensors")
            if not os.path.exists(weights_file):
                return
            from safetensors.torch import load_file

            result.load_state_dict(load_file(weights_file), strict=False)
        except Exception as e:  # noqa: BLE001 -- never break loading on repair
            logger.warning(f"weight re-materialization skipped: {e}")
