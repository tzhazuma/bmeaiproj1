#!/usr/bin/env python3
"""Get Kaggle download redirect URL and print it for aria2c."""
import os, sys, json, urllib.request

os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')

# Read kaggle.json
with open(os.path.expanduser('~/.kaggle/kaggle.json')) as f:
    creds = json.load(f)

username = creds.get('username','')
key = creds.get('key','')

url = 'https://www.kaggle.com/api/v1/datasets/download/aiocta/brats2023-part-1'
req = urllib.request.Request(url)
# Kaggle uses Basic Auth for legacy keys
import base64
auth = base64.b64encode(f'{username}:{key}'.encode()).decode()
req.add_header('Authorization', f'Basic {auth}')
req.add_header('User-Agent', 'kaggle-api/1.6.0')

# We need to follow redirects manually to get the final URL
opener = urllib.request.build_opener(
    urllib.request.HTTPRedirectHandler()
)
try:
    resp = opener.open(req, timeout=30)
    print(resp.geturl())
except urllib.error.HTTPError as e:
    print(f'Error {e.code}: {e.reason}', file=sys.stderr)
    print(e.read().decode()[:500], file=sys.stderr)
    sys.exit(1)
