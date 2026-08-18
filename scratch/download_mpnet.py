import ssl
import urllib3
import requests
from sentence_transformers import SentenceTransformer

# Fix Windows SSL Verification Issues for HuggingFace/Stanford downloads
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_orig_send = requests.Session.send
def _patched_send(self, request, **kwargs):
    kwargs['verify'] = False
    return _orig_send(self, request, **kwargs)
requests.Session.send = _patched_send

print("Downloading all-mpnet-base-v2 on a single thread...")
model = SentenceTransformer("all-mpnet-base-v2")
print("Download complete and cached!")
