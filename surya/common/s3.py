import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, Any

import requests
from tqdm import tqdm

from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()

# Lock file expiration time in seconds (10 minutes)
LOCK_EXPIRATION = 600

# Global lock for state file operations
state_file_lock = Lock()

# Network timeout settings
REQUEST_TIMEOUT = (30, 300)  # (connect_timeout, read_timeout)
RETRY_BACKOFF_MAX = 60  # Maximum retry delay in seconds


def get_file_status(download_dir: Path, filename: str) -> str:
    """Determine the download status of a file based on file existence.
    
    Args:
        download_dir: Directory where files are downloaded
        filename: Target filename
        
    Returns:
        'completed': File fully downloaded
        'in_progress': File is being/was being downloaded
        'not_started': File not downloaded yet
    """
    final_path = download_dir / filename
    progress_path = download_dir / f"{filename}.in-progress"
    
    if final_path.exists():
        return 'completed'
    elif progress_path.exists():
        return 'in_progress'
    else:
        return 'not_started'


def check_disk_space(path: Path, required_bytes: int, safety_margin: float = 0.1) -> bool:
    """Check if there's enough disk space for download.
    
    Args:
        path: Directory path to check
        required_bytes: Number of bytes needed
        safety_margin: Additional space margin (default: 10%)
        
    Returns:
        True if there's enough space, False otherwise
    """
    try:
        stat = os.statvfs(str(path))
        available_bytes = stat.f_bavail * stat.f_frsize
        required_with_margin = required_bytes * (1 + safety_margin)
        
        if available_bytes < required_with_margin:
            logger.error(f"Insufficient disk space. Required: {required_with_margin/1024/1024/1024:.2f}GB, "
                        f"Available: {available_bytes/1024/1024/1024:.2f}GB")
            return False
        return True
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True  # Assume OK if we can't check


def join_urls(url1: str, url2: str) -> str:
    """Join two URL parts safely.
    
    Args:
        url1: Base URL
        url2: Path to append
        
    Returns:
        Combined URL with proper separator
    """
    url1 = url1.rstrip("/")
    url2 = url2.lstrip("/")
    return f"{url1}/{url2}"


def get_model_name(pretrained_model_name_or_path: str) -> str:
    """Extract model name from model path.
    
    Args:
        pretrained_model_name_or_path: Full model path
        
    Returns:
        Model name (first part of path)
    """
    return pretrained_model_name_or_path.split("/")[0]


def download_file_with_resume(remote_path: str, local_path: str, chunk_size: int = 1024 * 1024, max_retries: int = 3) -> Path:
    """Download file with resume support and retry logic using temporary .in-progress files.
    
    Features:
    - Uses .in-progress suffix during download for reliable state tracking
    - Resume partial downloads using HTTP Range requests
    - Exponential backoff retry with timeout
    - Progress bar with file size information
    - Atomic rename on completion
    
    Args:
        remote_path: URL to download from
        local_path: Local file path to save to (final name)
        chunk_size: Size of download chunks in bytes (default: 1MB)
        max_retries: Maximum number of retry attempts (default: 3)
        
    Returns:
        Path object of the final downloaded file
        
    Raises:
        requests.exceptions.RequestException: For network-related errors
        OSError: For file system errors
        ValueError: For file integrity check failures
    """
    final_path = Path(local_path)
    progress_path = Path(f"{local_path}.in-progress")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If final file already exists, return it (already completed)
    if final_path.exists():
        logger.info(f"File {final_path.name} already exists, skipping download")
        return final_path
    
    for attempt in range(max_retries):
        pbar = None
        try:
            # Check if progress file partially exists
            resume_pos = 0
            if progress_path.exists():
                resume_pos = progress_path.stat().st_size
                logger.info(f"Resuming download from position {resume_pos} for {final_path.name}")
            
            # Prepare headers for resume
            headers = {}
            if resume_pos > 0:
                headers['Range'] = f'bytes={resume_pos}-'
            
            response = requests.get(
                remote_path, 
                stream=True, 
                allow_redirects=True, 
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            # Get total file size with better error handling
            total_size = 0
            if resume_pos > 0 and response.status_code == 206:  # Partial content
                content_range = response.headers.get('content-range', '')
                if content_range and '/' in content_range:
                    try:
                        total_size = int(content_range.split('/')[-1])
                    except (ValueError, IndexError):
                        logger.warning(f"Failed to parse content-range header: {content_range}")
                        total_size = resume_pos + int(response.headers.get('content-length', 0))
                else:
                    total_size = resume_pos + int(response.headers.get('content-length', 0))
            else:
                total_size = int(response.headers.get('content-length', 0))
                if response.status_code != 206:  # Server doesn't support resume
                    resume_pos = 0
                    if progress_path.exists():
                        progress_path.unlink()
            
            # Create progress bar
            filename = final_path.name
            pbar = tqdm(
                total=total_size,
                initial=resume_pos,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {filename}",
                miniters=1
            )
            
            # Download to progress file with resume
            mode = 'ab' if resume_pos > 0 else 'wb'
            with open(progress_path, mode) as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            pbar.close()
            pbar = None  # Mark as closed
            
            # Verify file size if known
            if total_size > 0:
                actual_size = progress_path.stat().st_size
                if actual_size != total_size:
                    raise ValueError(f"File size mismatch: expected {total_size}, got {actual_size}")
            
            # Atomic rename: download completed successfully
            progress_path.rename(final_path)
            
            logger.info(f"Successfully downloaded {filename}")
            return final_path
            
        except KeyboardInterrupt:
            logger.info(f"Download interrupted by user for {final_path.name}")
            logger.info(f"Progress file {progress_path.name} preserved for resume")
            raise  # Re-raise to terminate program
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error on attempt {attempt + 1}/{max_retries} for {remote_path}: {str(e)}")
            if attempt == max_retries - 1:
                # On final failure, clean up progress file
                if progress_path.exists():
                    progress_path.unlink()
                raise
            else:
                retry_delay = min(2 ** attempt, RETRY_BACKOFF_MAX)
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        except (OSError, IOError) as e:
            logger.error(f"File system error on attempt {attempt + 1}/{max_retries} for {final_path}: {str(e)}")
            if attempt == max_retries - 1:
                if progress_path.exists():
                    progress_path.unlink()
                raise
            else:
                retry_delay = min(2 ** attempt, RETRY_BACKOFF_MAX)
                time.sleep(retry_delay)
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}/{max_retries} for {remote_path}: {str(e)}")
            if attempt == max_retries - 1:
                if progress_path.exists():
                    progress_path.unlink()
                raise
            else:
                retry_delay = min(2 ** attempt, RETRY_BACKOFF_MAX)
                time.sleep(retry_delay)
        finally:
            # Ensure progress bar is always closed
            if pbar is not None:
                pbar.close()
    
    return final_path


def download_file(remote_path: str, local_path: str, chunk_size: int = 1024 * 1024) -> Path:
    """Legacy download function for backward compatibility.
    
    This function maintains the original API while providing enhanced
    resume support and error handling internally.
    
    Args:
        remote_path: URL to download from
        local_path: Local file path to save to
        chunk_size: Size of download chunks in bytes (default: 1MB)
        
    Returns:
        Path object of the downloaded file
    """
    return download_file_with_resume(remote_path, local_path, chunk_size)


def check_manifest(local_dir: str) -> bool:
    """Check if all files listed in manifest.json exist in the directory.
    
    Args:
        local_dir: Directory to check for complete download
        
    Returns:
        True if manifest exists and all files are present, False otherwise
    """
    local_dir = Path(local_dir)
    manifest_path = local_dir / "manifest.json"
    if not manifest_path.exists():
        return False

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        for file in manifest["files"]:
            if not (local_dir / file).exists():
                return False
    except Exception:
        return False

    return True




def download_directory(remote_path: str, local_dir: str) -> None:
    """Download an entire directory from S3 with resume support and robust error handling.
    
    This function provides enhanced download capabilities including:
    - Resume interrupted downloads from where they left off
    - Individual file retry with exponential backoff
    - Disk space verification before download
    - Thread-safe state tracking for concurrent operations  
    - Atomic file operations to prevent corruption
    - Progress reporting and detailed logging
    
    The function maintains backward compatibility with the original API while
    providing significant improvements in reliability and user experience.
    
    Args:
        remote_path: S3 path (without s3:// prefix) to download from
        local_dir: Local directory to save files to
        
    Raises:
        Exception: If download fails after all retries or insufficient disk space
        requests.exceptions.RequestException: For network-related errors
        OSError: For file system errors
    """
    model_name = get_model_name(remote_path)
    s3_url = join_urls(settings.S3_BASE_URL, remote_path)
    local_dir = Path(local_dir)
    
    # Check to see if it's already downloaded
    model_exists = check_manifest(local_dir)
    if model_exists:
        # Clean up any leftover download files if model is complete
        download_dir = local_dir / ".download"
        if download_dir.exists():
            shutil.rmtree(download_dir, ignore_errors=True)
        return

    # Create persistent download directory
    download_dir = local_dir / ".download"
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # Note: We deliberately do NOT check if download directory has completed files
    # This is important - we only move files after ALL downloads are confirmed successful
    # within the main download logic below. This ensures atomicity and consistency.
    # 
    # With .in-progress file naming, we no longer need complex state file tracking
    # as file status is determined directly from file existence.
    
    try:
        # Download the manifest file first
        manifest_file = join_urls(s3_url, "manifest.json")
        manifest_path = download_dir / "manifest.json"
        if not manifest_path.exists():
            download_file(manifest_file, str(manifest_path))

        # Load manifest
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Print file list for debugging/external download tools
        logger.info(f"Files to download for model {model_name}:")
        total_files = len(manifest["files"])
        for i, file in enumerate(manifest["files"], 1):
            remote_file_url = join_urls(s3_url, file)
            logger.info(f"  [{i}/{total_files}] {file} -> {remote_file_url}")

        # Determine which files need downloading using simple file status check
        files_to_download = []
        
        logger.info("Checking file download status...")
        for file in manifest["files"]:
            file_status = get_file_status(download_dir, file)
            
            if file_status == 'completed':
                logger.debug(f"File {file} already completed")
                continue
            elif file_status == 'in_progress':
                logger.info(f"File {file} partially downloaded, will resume")
                files_to_download.append(file)
            else:  # not_started
                logger.debug(f"File {file} not started")
                files_to_download.append(file)

        if not files_to_download:
            logger.info("All files already downloaded, moving to final location...")
        else:
            logger.info(f"Need to download {len(files_to_download)} files (out of {total_files} total)")
            
            # Check disk space before starting downloads
            try:
                total_size_needed = 0
                logger.info("Checking file sizes and disk space...")
                
                for file in files_to_download[:5]:  # Check first 5 files for size estimation
                    try:
                        remote_file = join_urls(s3_url, file)
                        response = requests.head(remote_file, timeout=REQUEST_TIMEOUT)
                        if response.status_code == 200:
                            file_size = int(response.headers.get('content-length', 0))
                            total_size_needed += file_size
                    except Exception:
                        # If we can't check size, estimate 100MB per file
                        total_size_needed += 100 * 1024 * 1024
                
                # Extrapolate for all files
                if files_to_download:
                    avg_size = total_size_needed / min(len(files_to_download), 5)
                    estimated_total = avg_size * len(files_to_download)
                    
                    logger.info(f"Estimated download size: {estimated_total/1024/1024/1024:.2f}GB")
                    
                    if not check_disk_space(local_dir, int(estimated_total)):
                        raise Exception("Insufficient disk space for download")
                        
            except Exception as e:
                if "Insufficient disk space" in str(e):
                    raise
                logger.warning(f"Could not verify disk space: {e}")

            # Download remaining files with individual retry logic
            pbar = None
            try:
                pbar = tqdm(
                    desc=f"Downloading {model_name} model",
                    total=len(files_to_download),
                )

                # Use limited parallelism to avoid overwhelming the network
                max_workers = min(settings.PARALLEL_DOWNLOAD_WORKERS, 2)  # Limit concurrent downloads
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    def download_single_file(file):
                        remote_file = join_urls(s3_url, file)
                        local_file = download_dir / file
                        try:
                            download_file_with_resume(remote_file, str(local_file), max_retries=3)
                            return file, True, None
                        except Exception as e:
                            return file, False, str(e)

                    # Submit all download tasks
                    future_to_file = {executor.submit(download_single_file, file): file for file in files_to_download}
                    
                    # Process completed downloads using as_completed for better error handling
                    failed_downloads = []
                    
                    for future in as_completed(future_to_file):
                        file = future_to_file[future]
                        try:
                            file, success, error = future.result()
                            
                            if success:
                                logger.debug(f"Successfully downloaded {file}")
                            else:
                                logger.error(f"Failed to download {file}: {error}")
                                failed_downloads.append(file)
                            
                        except Exception as e:
                            logger.error(f"Unexpected error processing download result for {file}: {e}")
                            failed_downloads.append(file)
                        
                        pbar.update(1)
                        
            finally:
                if pbar is not None:
                    pbar.close()

        # Check if any files failed
        if failed_downloads:
            raise Exception(f"Failed to download {len(failed_downloads)} files: {failed_downloads}")

        # All files downloaded successfully, move to final location
        logger.info("Moving downloaded files to final location...")
        
        # Move manifest.json first
        manifest_src = download_dir / "manifest.json"
        manifest_dst = local_dir / "manifest.json"
        if manifest_src.exists():
            shutil.move(str(manifest_src), str(manifest_dst))
        
        # Move all other files
        for file in manifest["files"]:
            src_path = download_dir / file
            dst_path = local_dir / file
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))

        # Clean up download directory
        shutil.rmtree(download_dir, ignore_errors=True)
        
        logger.info(f"Successfully downloaded model {model_name}")

    except KeyboardInterrupt:
        logger.info(f"Download interrupted by user for model {model_name}")
        logger.info(f"Download directory {download_dir} preserved for resume")
        raise  # Re-raise to terminate program
    except Exception as e:
        logger.error(f"Download failed for model {model_name}: {e}")
        # Keep download directory for resume
        raise


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

        return super().from_pretrained(local_path, *args, **kwargs)
