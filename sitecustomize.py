# Project local sitecustomize to suppress specific noisy warnings in dev
import warnings
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings('ignore', category=NotOpenSSLWarning)
except Exception:
    pass
