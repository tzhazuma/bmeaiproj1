#!/usr/bin/env python3
"""
BraTS2023 Dataset Downloader
Supports Kaggle API with automatic authentication.
Usage:
    python scripts/download_brats.py --output /mnt/d/brats2023
"""
import os
import sys
import argparse
import json
import urllib.request
import base64


def get_kaggle_credentials():
    """Read Kaggle credentials from ~/.kaggle/kaggle.json"""
    kaggle_json = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(kaggle_json):
        raise FileNotFoundError(
            f"Kaggle credentials not found at {kaggle_json}.\n"
            "Please download your API token from https://www.kaggle.com/settings/account "
            "and place it at ~/.kaggle/kaggle.json"
        )
    with open(kaggle_json) as f:
        return json.load(f)


def get_download_url(creds, dataset='aiocta/brats2023-part-1'):
    """Get signed GCS download URL from Kaggle API."""
    url = f'https://www.kaggle.com/api/v1/datasets/download/{dataset}'
    req = urllib.request.Request(url)
    username = creds.get('username', '')
    key = creds.get('key', '')
    auth = base64.b64encode(f'{username}:{key}'.encode()).decode()
    req.add_header('Authorization', f'Basic {auth}')
    req.add_header('User-Agent', 'kaggle-api/1.6.0')
    opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
    resp = opener.open(req, timeout=30)
    return resp.geturl()


def download_with_aria2c(url, output_path, max_retries=3):
    """Download using aria2c with multi-threading and resume support."""
    import subprocess
    output_dir = os.path.dirname(os.path.abspath(output_path))
    output_name = os.path.basename(output_path)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        'aria2c', '-x', '16', '-s', '16', '-c',
        '--file-allocation=none',
        '-d', output_dir,
        '-o', output_name,
        url
    ]
    for attempt in range(max_retries):
        print(f"Download attempt {attempt + 1}/{max_retries}...")
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print(f"Download complete: {output_path}")
            return True
        print(f"Attempt {attempt + 1} failed, retrying...")
    return False


def extract_dataset(zip_path, extract_to):
    """Extract ZIP dataset with progress."""
    import zipfile
    from tqdm import tqdm

    print(f"Extracting {zip_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        members = z.namelist()
        for member in tqdm(members, desc='Extracting'):
            z.extract(member, extract_to)
    print("Extraction complete!")


def main():
    parser = argparse.ArgumentParser(description='Download BraTS2023 dataset')
    parser.add_argument('--output', default='/mnt/d/brats2023', help='Output directory')
    parser.add_argument('--dataset', default='aiocta/brats2023-part-1', help='Kaggle dataset ref')
    parser.add_argument('--skip-download', action='store_true', help='Skip download if zip exists')
    parser.add_argument('--skip-extract', action='store_true', help='Skip extraction')
    args = parser.parse_args()

    zip_path = os.path.join(args.output, 'brats2023-part-1.zip')

    if not args.skip_download or not os.path.exists(zip_path):
        print("Getting Kaggle download URL...")
        creds = get_kaggle_credentials()
        url = get_download_url(creds, args.dataset)
        print(f"Download URL obtained, starting download...")
        if not download_with_aria2c(url, zip_path):
            print("Download failed after retries.")
            sys.exit(1)
    else:
        print(f"Using existing zip: {zip_path}")

    if not args.skip_extract:
        extract_dataset(zip_path, args.output)
        # Verify
        patients = [d for d in os.listdir(args.output)
                    if os.path.isdir(os.path.join(args.output, d)) and d.startswith('BraTS')]
        print(f"Dataset ready. Patient folders: {len(patients)}")


if __name__ == '__main__':
    main()
