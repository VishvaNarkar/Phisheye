# utils.py
import re, email, base64
from bs4 import BeautifulSoup

URL_REGEX = re.compile(
    r'((?:https?://|www\.)[^\s<>")]+|(?:mailto:)[^\s<>")]+)', re.IGNORECASE
)

def basic_clean(x: str) -> str:
    # strip HTML, normalize spaces
    soup = BeautifulSoup(x, "lxml")
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_urls_from_text(text: str):
    return URL_REGEX.findall(text)

def parse_eml_bytes(raw: bytes) -> str:
    msg = email.message_from_bytes(raw)
    parts = []
    if msg.is_multipart():
        for p in msg.walk():
            ctype = p.get_content_type()
            if ctype in ["text/plain","text/html"]:
                try:
                    parts.append(p.get_payload(decode=True).decode(p.get_content_charset() or "utf-8", "ignore"))
                except Exception:
                    pass
    else:
        payload = msg.get_payload(decode=True) or b""
        try:
            parts.append(payload.decode(msg.get_content_charset() or "utf-8", "ignore"))
        except Exception:
            pass
    return "\n".join(parts)
