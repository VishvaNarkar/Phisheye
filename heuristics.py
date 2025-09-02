# heuristics.py
import re
from urllib.parse import urlparse
from utils import extract_urls_from_text

SUSPICIOUS_WORDS = [
    "verify", "urgent", "suspend", "login", "credential", "password",
    "click here", "confirm", "limited time", "update account", "reset"
]

def url_red_flags(url: str):
    flags = []
    p = urlparse(url if url.startswith(("http","mailto")) else "http://" + url)
    host = (p.netloc or p.path).lower()

    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host):
        flags.append("IP-based URL")
    if host.count("-") >= 2:
        flags.append("Many hyphens in domain")
    if host.endswith((".zip",".mov",".top",".xyz",".ru",".cn",".buzz",".rest",".click")):
        flags.append("Suspicious TLD")
    if "@" in host or ".." in host:
        flags.append("Malformed domain")
    if not p.scheme or p.scheme == "http":
        flags.append("Not HTTPS")
    return flags

def text_red_flags(text: str):
    t = text.lower()
    out = []
    for w in SUSPICIOUS_WORDS:
        if w in t:
            out.append(f'Contains "{w}"')
    # ask for personal data or payment
    if any(k in t for k in ["ssn","otp","cvv","card number","bank account"]):
        out.append("Requests sensitive info")
    return out

def run_heuristics(text: str):
    urls = extract_urls_from_text(text)
    findings = {"urls": [], "content_flags": text_red_flags(text)}
    for u in urls:
        findings["urls"].append({"url": u, "flags": url_red_flags(u)})
    # score = small bump per flag
    score = 0.0
    score += 0.1 * len(findings["content_flags"])
    score += 0.15 * sum(len(x["flags"]) for x in findings["urls"])
    score = min(score, 0.9)
    return findings, score
