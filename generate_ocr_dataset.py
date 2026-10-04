"""Synthetic OCR line-crop generator for PaddleOCR-style recognizer training.

The generator takes real text (Wikipedia XML dumps, JSONL article collections,
or an existing label manifest), renders each text chunk into a real browser font
stack, applies a configurable scanner/photocopy degradation pipeline, and writes
PaddleOCR-ready training manifests plus a rich per-row metadata journal.

Design notes
------------
* Text is rendered by headless Chrome/Edge, not by a text rasterizer. That means
  real font binaries, real CSS layout, real Arabic-script shaping and bidi, and
  real ligature/kerning behaviour.
* Rendering is batched. Many independent text lines are placed on one tall
  "sheet" page and captured in a single screenshot, which is far faster than one
  browser launch per line. Failed sheets are automatically bisected into smaller
  screenshots and retried.
* Every degradation decision is driven by a per-line seeded RNG, so a given
  ``--seed`` reproduces byte-identical crops regardless of worker scheduling.
* Output is fail-closed: rows are only committed to the manifests after the
  image bytes have been written, and resume state is validated before appending.

See README.md for a full walkthrough.
"""

import argparse
import bz2
import hashlib
import html
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, features

try:
    import mwparserfromhell
except ImportError:
    mwparserfromhell = None

# Arabic-script normalization is much stronger when the optional `asosoft`
# Kurdish text engine is installed. It is not on PyPI, so the generator runs
# without it and simply skips the extra normalization passes.
try:
    import asosoft
except Exception:
    asosoft = None

# The generator already parallelises across threads. Letting OpenCV also spin up
# its own thread pool oversubscribes the CPU and slows the whole run down.
cv2.setNumThreads(1)

__version__ = "1.0.0"


def numpy_rng(rng: random.Random) -> np.random.Generator:
    """Return a NumPy generator seeded from a Python RNG.

    Degradation code needs vectorised noise. Pulling it from the global NumPy
    RNG would make results depend on how many crops happened to be processed
    before this one, which breaks reproducibility under parallel workers.
    Deriving a local generator from the caller's seeded RNG keeps every crop
    independent and reproducible.
    """
    return np.random.default_rng(rng.getrandbits(63))


# Conventional Wikipedia dump file names, matched by glob inside
# --default-sources-dir. Only MediaWiki dumps are auto-discovered; anything else
# must be passed explicitly with --source, because guessing a file's format from
# its name is not reliable.
#
# Wikimedia publishes dumps as e.g.
#   fawiki-latest-pages-articles1.xml-p1500001p3000000.bz2
# so these patterns match on the prefix and the .bz2 suffix rather than assuming
# a plain ".xml.bz2" ending.
DEFAULT_SOURCE_PATTERNS = [
    ("*wiki-latest-pages-articles*.xml.bz2", "wikipedia"),
    ("*wiki-latest-pages-articles*.xml", "wikipedia"),
]

# Recognised values for the `:type` suffix of --source.
#
# These describe file *formats*, not particular websites or corpora. Bring your
# own text: point --source at any file in one of these shapes and it will be
# read.
SOURCE_TYPES = {
    "wikipedia": "MediaWiki XML dump (plain or .bz2). The usual choice.",
    "wikitext": "A single file of raw MediaWiki wikitext markup.",
    "jsonl": "JSON Lines, one article per line. Common field names such as "
             "title/headline, summary/description/excerpt, and "
             "content/text/body/article_html/content_html are picked up automatically.",
    "text": "Any plain text file. The whole file is treated as one document.",
}

# Wikipedia dumps anyone can download directly from Wikimedia. These are public
# archive URLs, not bundled data: the script never ships a corpus.
WIKI_DUMP_URLS = {
    "fawiki": "https://dumps.wikimedia.org/fawiki/latest/fawiki-latest-pages-articles1.xml-p1500001p3000000.bz2",
    "ckbwiki": "https://dumps.wikimedia.org/ckbwiki/latest/ckbwiki-latest-pages-articles.xml.bz2",
    "kuwiki": "https://dumps.wikimedia.org/kuwiki/latest/kuwiki-latest-pages-articles.xml.bz2",
    "enwiki": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-p1p1000.bz2",
    "arwiki": "https://dumps.wikimedia.org/arwiki/latest/arwiki-latest-pages-articles1.xml-p1p1000.bz2",
}


def discover_default_sources(directory: Path) -> list[tuple[str, str]]:
    """Return ``(path, source_type)`` pairs for recognised files in ``directory``.

    A missing directory is not an error: --use-default-sources is a convenience,
    and explicit --source flags remain the supported way to point at corpora.
    """
    if not directory.is_dir():
        return []
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pattern, kind in DEFAULT_SOURCE_PATTERNS:
        for path in sorted(directory.glob(pattern)):
            if not path.is_file():
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append((key, kind))
    return found


CONSERVATIVE_SORANI_MAP = {
    "\u0643": "\u06a9",  # ARABIC KAF -> ARABIC KEHEH
    "\u0649": "\u06cc",  # ALEF MAKSURA -> FARSI YEH
    "\u061c": "",  # ARABIC LETTER MARK
    "\u200e": "",  # LEFT-TO-RIGHT MARK
    "\u200f": "",  # RIGHT-TO-LEFT MARK
    "\ufeff": "",  # BOM / ZERO WIDTH NO-BREAK SPACE
}

ARABIC_MARK_RANGES = (
    (0x064B, 0x065F),
    (0x0670, 0x0670),
    (0x06D6, 0x06ED),
)

NON_TATWEEL_EXTENDING_ARABIC_LETTERS = set(
    "\u0621\u0622\u0623\u0624\u0625\u0627\u0629"
    "\u062f\u0630\u0631\u0632"
    "\u0648\u0649"
    "\u0671\u0672\u0673\u0675"
    "\u0688\u0689\u068a\u068b\u068c\u068d\u068e\u068f\u0690"
    "\u0691\u0692\u0693\u0694\u0695\u0696\u0697\u0698\u0699\u069a"
    "\u06c0\u06c4\u06c5\u06c6\u06c7\u06c8\u06c9\u06ca\u06cb\u06d5"
)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")




def is_arabic_mark(char: str) -> bool:
    codepoint = ord(char)
    return any(lo <= codepoint <= hi for lo, hi in ARABIC_MARK_RANGES)


def is_arabic_letter(char: str) -> bool:
    return unicodedata.name(char, "").startswith("ARABIC LETTER ")


def can_extend_to_following_tatweel(char: str) -> bool:
    return is_arabic_letter(char) and char not in NON_TATWEEL_EXTENDING_ARABIC_LETTERS


def next_base_char(text: str, index: int) -> str | None:
    for char in text[index + 1 :]:
        if unicodedata.category(char).startswith("M") or char == "\u0640":
            continue
        return char
    return None


def previous_base_char(text: str, index: int) -> str | None:
    for char in reversed(text[:index]):
        if unicodedata.category(char).startswith("M") or char == "\u0640":
            continue
        return char
    return None


def should_normalize_arabic_yeh_nonfinal(text: str, index: int) -> bool:
    next_char = next_base_char(text, index)
    return next_char is not None and is_arabic_letter(next_char)


def normalize_tatweel_contextually(text: str, index: int) -> str:
    previous_char = previous_base_char(text, index)
    next_char = next_base_char(text, index)
    previous_can_extend = previous_char is not None and can_extend_to_following_tatweel(previous_char)
    next_is_arabic = next_char is not None and is_arabic_letter(next_char)

    if previous_can_extend and not next_is_arabic:
        return "\u0640"
    if not previous_can_extend and not next_is_arabic:
        return "-"
    return ""


def normalize_ocr_label_text(
    text: str,
    *,
    profile: str,
    strip_arabic_marks: bool,
    normalize_arabic_yeh_nonfinal: bool,
    normalize_zwnj: bool,
) -> str:
    if not text:
        return ""
    if profile == "none":
        mapping: dict[str, str] = {}
        normalize_tatweel = False
    elif profile == "conservative_sorani":
        mapping = dict(CONSERVATIVE_SORANI_MAP)
        normalize_tatweel = True
    else:
        raise ValueError(f"unknown text cleanup profile: {profile}")

    if normalize_zwnj:
        mapping["\u200c"] = ""

    out = []
    for index, char in enumerate(text):
        category = unicodedata.category(char)
        if strip_arabic_marks and is_arabic_mark(char):
            continue
        if category == "Cf" and char not in {"\u200c"}:
            continue
        if normalize_tatweel and char == "\u0640":
            out.append(normalize_tatweel_contextually(text, index))
            continue
        if normalize_arabic_yeh_nonfinal and char == "\u064a" and should_normalize_arabic_yeh_nonfinal(text, index):
            out.append("\u06cc")
            continue
        out.append(mapping.get(char, char))
    cleaned = "".join(out)
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_text(
    text: str,
    use_asosoft: bool = True,
    cleanup_profile: str = "conservative_sorani",
    strip_arabic_marks: bool = False,
    normalize_arabic_yeh_nonfinal: bool = False,
    normalize_zwnj: bool = False,
) -> str:
    if not text:
        return ""
    if use_asosoft and asosoft is not None:
        try:
            text = asosoft.Normalize(text)
        except Exception:
            pass
    return normalize_ocr_label_text(
        text,
        profile=cleanup_profile,
        strip_arabic_marks=strip_arabic_marks,
        normalize_arabic_yeh_nonfinal=normalize_arabic_yeh_nonfinal,
        normalize_zwnj=normalize_zwnj,
    )


def convert_kmr_to_arabic(text: str) -> str:
    if not text:
        return ""
    if asosoft is None:
        return text
    try:
        return asosoft.La2Ar(text)
    except Exception:
        return text


def clean_html(raw_html: str, use_asosoft: bool = True, **cleanup_kwargs) -> str:
    text = html.unescape(raw_html or "")
    text = re.sub(r"</(p|div|br|li|h[1-6]|article|section|blockquote)>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_text(text, use_asosoft=use_asosoft, **cleanup_kwargs)


def join_unique_text_blocks(*blocks: str, use_asosoft: bool = True, **cleanup_kwargs) -> str:
    seen: set[str] = set()
    kept: list[str] = []
    for block in blocks:
        cleaned = normalize_text(block or "", use_asosoft=use_asosoft, **cleanup_kwargs)
        key = re.sub(r"\s+", " ", cleaned).strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(cleaned)
    return "\n\n".join(kept)


# Language tags for which the Kurdish-specific text normalizer must not run.
# `asosoft` rewrites Arabic-script text toward Kurdish orthography, so applying it
# to Arabic, Persian, Turkish or English records would corrupt their labels.
NON_KURDISH_LANGUAGE_TAGS = {
    "en",
    "eng",
    "english",
    "ar",
    "ara",
    "arabic",
    "fa",
    "fas",
    "per",
    "persian",
    "farsi",
    "tr",
    "tur",
    "turkish",
    "ku",
    "kur",
    "kmr",
    "kurmanji",
    "ckb",
    "srd",
    "ku-ar",
    "ku-latn",
}


def should_use_asosoft(language: str) -> bool:
    """Decide whether the Kurdish-specific normalizer applies to a record.

    Driven purely by the record's declared language tag, so it works for any
    corpus rather than for a fixed list of known websites.
    """
    language = (language or "").strip().lower()
    if not language:
        return True
    return language not in NON_KURDISH_LANGUAGE_TAGS


def clean_wikitext(text: str, **cleanup_kwargs) -> str:
    if not text:
        return ""
    if mwparserfromhell is not None:
        try:
            code = mwparserfromhell.parse(text)
            text = code.strip_code(normalize=True, collapse=True)
        except Exception:
            return ""
    else:
        text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
        text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"'{2,}", "", text)
    return normalize_text(text, **cleanup_kwargs)


# Field names commonly used for each part of an article in JSONL exports. The
# streamer picks the first one present, so most JSONL corpora work unchanged
# without needing a per-corpus reader.
JSONL_TITLE_FIELDS = ("title", "headline", "heading", "name")
JSONL_SUMMARY_FIELDS = ("summary", "description", "excerpt", "subtitle", "lead", "abstract")
JSONL_BODY_HTML_FIELDS = ("article_html", "content_html", "body_html", "html")
JSONL_BODY_TEXT_FIELDS = (
    "content_text",
    "content",
    "body",
    "text",
    "article",
    "raw_content",
)
JSONL_LANGUAGE_FIELDS = ("language", "lang", "locale", "language_code")
JSONL_ID_FIELDS = ("id", "uuid", "url", "uri", "link", "slug")


class TextStreamer:
    """Streams (title, text) documents out of a text file.

    The reader is chosen by the ``:type`` suffix given to --source, which
    describes the file format rather than where the text came from. No corpus,
    website or dataset is referenced by name anywhere in here.
    """

    def __init__(
        self,
        pseudo_kurdish: bool,
        max_seen: int,
        cleanup_profile: str,
        strip_arabic_marks: bool,
        normalize_arabic_yeh_nonfinal: bool,
        normalize_zwnj: bool,
        convert_latin_kurdish_to_arabic: bool = False,
    ):
        from collections import deque

        self.pseudo_kurdish = pseudo_kurdish
        self.convert_latin_kurdish_to_arabic = convert_latin_kurdish_to_arabic
        self.cleanup_kwargs = {
            "cleanup_profile": cleanup_profile,
            "strip_arabic_marks": strip_arabic_marks,
            "normalize_arabic_yeh_nonfinal": normalize_arabic_yeh_nonfinal,
            "normalize_zwnj": normalize_zwnj,
        }
        self._seen_deque = deque(maxlen=max_seen)
        self._seen_set = set()

    def _maybe_shuffle_words(self, text: str) -> str:
        if not self.pseudo_kurdish:
            return text
        paragraphs = text.split("\n")
        out = []
        for paragraph in paragraphs:
            words = [w for w in paragraph.split() if w.strip()]
            if words:
                random.shuffle(words)
                out.append(" ".join(words))
        return "\n\n".join(out)

    def _stream_wikipedia_xml(self, filepath: str) -> Iterable[tuple[str, str]]:
        opener = bz2.open if filepath.endswith(".bz2") else open
        with opener(filepath, "rt", encoding="utf-8", errors="replace") as f:
            for _, elem in ET.iterparse(f, events=("end",)):
                if not elem.tag.endswith("page"):
                    continue
                title = ""
                raw_text = ""
                redirect = False
                for child in elem:
                    tag = child.tag.split("}")[-1]
                    if tag == "title":
                        title = child.text or ""
                    elif tag == "redirect":
                        redirect = True
                    elif tag == "revision":
                        for item in child:
                            if item.tag.split("}")[-1] == "text":
                                raw_text = item.text or ""
                # Release the element immediately. iterparse keeps every parsed
                # element alive, and a full dump does not fit in memory.
                elem.clear()
                if redirect:
                    continue
                body = clean_wikitext(raw_text, **self.cleanup_kwargs)
                if len(body) < 100:
                    continue
                full = normalize_text(f"{title}\n\n{body}", **self.cleanup_kwargs)
                if self.convert_latin_kurdish_to_arabic:
                    title = convert_kmr_to_arabic(title)
                    full = convert_kmr_to_arabic(full)
                yield title, full

    def _stream_jsonl(self, filepath: str) -> Iterable[tuple[str, str]]:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue

                doc_id = next((str(data[k]) for k in JSONL_ID_FIELDS if data.get(k)), None)
                if doc_id is None:
                    doc_id = hashlib.sha1(line.encode("utf-8", "replace")).hexdigest()
                if doc_id in self._seen_set:
                    continue
                if len(self._seen_deque) == self._seen_deque.maxlen:
                    self._seen_set.discard(self._seen_deque[0])
                self._seen_deque.append(doc_id)
                self._seen_set.add(doc_id)

                language = next((str(data[k]) for k in JSONL_LANGUAGE_FIELDS if data.get(k)), "")
                use_asosoft = should_use_asosoft(language)

                title = next((str(data[k]) for k in JSONL_TITLE_FIELDS if data.get(k)), "")
                summary = next((str(data[k]) for k in JSONL_SUMMARY_FIELDS if data.get(k)), "")
                html_body = next((data[k] for k in JSONL_BODY_HTML_FIELDS if data.get(k)), "")
                text_body = next((data[k] for k in JSONL_BODY_TEXT_FIELDS if data.get(k)), "")

                if html_body:
                    body = clean_html(str(html_body), use_asosoft=use_asosoft, **self.cleanup_kwargs)
                elif text_body:
                    body = normalize_text(str(text_body), use_asosoft=use_asosoft, **self.cleanup_kwargs)
                else:
                    continue

                full = join_unique_text_blocks(
                    title,
                    summary,
                    body,
                    use_asosoft=use_asosoft,
                    **self.cleanup_kwargs,
                )
                if self.convert_latin_kurdish_to_arabic:
                    title = convert_kmr_to_arabic(title)
                    full = convert_kmr_to_arabic(full)
                if len(full) <= 50:
                    continue
                yield title, full

    def _stream_wikitext(self, filepath: str) -> Iterable[tuple[str, str]]:
        opener = bz2.open if filepath.endswith(".bz2") else open
        with opener(filepath, "rt", encoding="utf-8", errors="replace") as f:
            full = clean_wikitext(f.read(), **self.cleanup_kwargs)
        if len(full) > 50:
            yield "", full

    def _stream_plain_text(self, filepath: str) -> Iterable[tuple[str, str]]:
        opener = bz2.open if filepath.endswith(".bz2") else open
        with opener(filepath, "rt", encoding="utf-8", errors="replace") as f:
            full = normalize_text(f.read(), **self.cleanup_kwargs)
        if len(full) > 50:
            yield "", full

    def stream_file(self, filepath: str, source_type: str, skip_count: int = 0) -> Iterable[tuple[str, str]]:
        readers = {
            "wikipedia": self._stream_wikipedia_xml,
            "jsonl": self._stream_jsonl,
            "wikitext": self._stream_wikitext,
            "text": self._stream_plain_text,
        }
        reader = readers.get(source_type)
        if reader is None:
            raise ValueError(f"unsupported source_type={source_type!r} for file={filepath}")

        yielded = 0
        for title, text in reader(filepath):
            text = self._maybe_shuffle_words(text)
            yielded += 1
            if yielded <= skip_count:
                continue
            yield title, text


@dataclass
class DegradeConfig:
    profile: str
    jpeg_quality_min: int
    jpeg_quality_max: int
    perspective_warp_prob: float
    max_perspective_shift_frac: float
    geometry_pad_px: int
    max_rotation_degrees: float
    scanner_band_prob: float
    edge_shadow_prob: float
    stain_prob: float
    extra_noise_prob: float
    nearby_rule_prob: float
    heavy_blur_prob: float
    low_quality_jpeg_prob: float
    low_quality_jpeg_min: int
    low_quality_jpeg_max: int
    photocopy_bleed_intensity_min: float
    photocopy_bleed_intensity_max: float
    binary_bloat_intensity_min: float
    binary_bloat_intensity_max: float


def add_paper_background(img: np.ndarray, rng: random.Random, strength: float) -> np.ndarray:
    h, w = img.shape[:2]
    base = img.astype(np.int16, copy=True)
    x_grad = np.linspace(rng.uniform(-10, 10), rng.uniform(-10, 10), w, dtype=np.float32)
    y_grad = np.linspace(rng.uniform(-8, 8), rng.uniform(-8, 8), h, dtype=np.float32)[:, None]
    noise = rng.uniform(0.0, strength) * numpy_rng(rng).normal(0, 4.0, img.shape[:2]).astype(np.float32)
    shade = np.rint(x_grad[None, :] + y_grad + noise).astype(np.int16)
    if base.ndim == 3:
        base += shade[:, :, None]
    else:
        base += shade
    np.clip(base, 0, 255, out=base)
    return base.astype(np.uint8, copy=False)


def add_full_image_noise(img: np.ndarray, rng: random.Random) -> np.ndarray:
    sigma = rng.uniform(1.6, 5.5)
    noise = np.rint(numpy_rng(rng).normal(0, sigma, img.shape)).astype(np.int16)
    out = img.astype(np.int16, copy=True)
    out += noise
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8, copy=False)


def add_scanner_band(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    freq = rng.uniform(0.015, 0.055)
    amp = rng.uniform(3.0, 9.0)
    angle = rng.uniform(0, np.pi)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    wave = amp * np.sin(2 * np.pi * freq * (xx * np.cos(angle) + yy * np.sin(angle)))
    out = img.astype(np.float32) + wave[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def add_edge_shadow(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    shade_w = max(2, rng.randint(max(2, w // 28), max(3, w // 10)))
    side = rng.choice(["left", "right"])
    alpha_max = rng.uniform(10.0, 36.0)
    shadow = np.zeros((h, w), dtype=np.float32)
    ramp = np.linspace(alpha_max, 0.0, shade_w, dtype=np.float32)
    if side == "left":
        shadow[:, :shade_w] = ramp[None, :]
    else:
        shadow[:, w - shade_w:] = ramp[::-1][None, :]
    out = img.astype(np.float32) - shadow[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def add_stains(img: np.ndarray, rng: random.Random) -> np.ndarray:
    pil_img = Image.fromarray(img).convert("RGBA")
    layer = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w, h = pil_img.size
    for _ in range(rng.randint(1, 2)):
        cx = rng.randint(0, max(1, w))
        cy = rng.randint(0, max(1, h))
        rx = rng.randint(max(4, w // 30), max(6, w // 10))
        ry = rng.randint(max(3, h // 6), max(4, h // 2))
        color = (
            rng.randint(130, 190),
            rng.randint(105, 165),
            rng.randint(55, 110),
            rng.randint(8, 28),
        )
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color)
    return np.array(Image.alpha_composite(pil_img, layer).convert("RGB"))


def add_nearby_rule_fragment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    pil_img = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size
    color = rng.choice([(0, 0, 0), (70, 70, 70), (135, 135, 135), (185, 185, 185)])
    width = rng.choice([1, 1, 2])
    if rng.random() < 0.70:
        y = rng.choice([rng.randint(0, max(0, h // 5)), rng.randint(max(0, h - h // 5), max(0, h - 1))])
        x0 = rng.randint(0, max(0, w // 6))
        x1 = rng.randint(max(x0 + 8, w * 2 // 3), w)
        draw.line((x0, y, x1, y), fill=color, width=width)
    else:
        x = rng.choice([rng.randint(0, max(0, w // 12)), rng.randint(max(0, w - w // 12), max(0, w - 1))])
        draw.line((x, 0, x, h), fill=color, width=width)
    return np.array(pil_img)


def add_photocopy_bleed_pixelation(img: np.ndarray, rng: random.Random, intensity: float) -> np.ndarray:
    intensity = max(0.0, min(1.0, intensity))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if rng.random() < 0.65:
        lo = int(162 + 18 * intensity)
        hi = int(196 + 28 * intensity)
        _, binary = cv2.threshold(gray, rng.randint(lo, hi), 255, cv2.THRESH_BINARY)
    else:
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            rng.choice([17, 21, 25, 31]),
            rng.randint(max(1, int(8 - 5 * intensity)), max(2, int(12 - 6 * intensity))),
        )

    ink = 255 - binary
    if intensity < 0.35:
        k = rng.choice([1, 2, 2])
        iterations = 1
    elif intensity < 0.70:
        k = rng.choice([2, 2, 3])
        iterations = rng.choice([1, 1, 2])
    else:
        k = rng.choice([2, 3, 3])
        iterations = rng.choice([1, 2, 2])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    ink = cv2.dilate(ink, kernel, iterations=iterations)

    h, w = ink.shape[:2]
    scale_hi = 0.82 - 0.24 * intensity
    scale_lo = 0.58 - 0.28 * intensity
    scale = rng.uniform(max(0.26, scale_lo), max(0.34, scale_hi))
    low_w = max(8, int(w * scale))
    low_h = max(8, int(h * scale))
    low = cv2.resize(ink, (low_w, low_h), interpolation=cv2.INTER_AREA)
    blocky = cv2.resize(low, (w, h), interpolation=cv2.INTER_NEAREST)

    if rng.random() < 0.25 + 0.35 * intensity:
        blocky = cv2.GaussianBlur(blocky, (3, 3), rng.uniform(0.10, 0.55 + 0.35 * intensity))
    out_gray = 255 - blocky
    out = cv2.cvtColor(out_gray, cv2.COLOR_GRAY2RGB)

    if rng.random() < 0.20 + 0.35 * intensity:
        out = add_full_image_noise(out, rng)
    return out


def add_binary_lowres_bloat(img: np.ndarray, rng: random.Random, intensity: float) -> np.ndarray:
    intensity = max(0.0, min(1.0, intensity))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape[:2]
    pre_scale = (0.94 - 0.48 * intensity) + rng.uniform(-0.020, 0.020)
    pre_scale = max(0.42, min(0.96, pre_scale))
    low_w = max(8, int(w * pre_scale))
    low_h = max(8, int(h * pre_scale))
    low = cv2.resize(gray, (low_w, low_h), interpolation=cv2.INTER_AREA)

    if intensity >= 0.18:
        low = cv2.GaussianBlur(low, (3, 3), rng.uniform(0.05, 0.18 + 0.28 * intensity))

    dark_pixels = low[low < np.percentile(low, 84)]
    if dark_pixels.size:
        base_threshold = int(np.percentile(dark_pixels, 84 + 10 * intensity))
    else:
        base_threshold = int(np.percentile(low, 35))
    threshold = max(80, min(235, base_threshold + int(round(5 + 34 * intensity)) + rng.randint(-2, 3)))
    _, binary_low = cv2.threshold(low, threshold, 255, cv2.THRESH_BINARY)

    ink = 255 - binary_low

    bridge_w = 1 + int(round(8 * intensity))
    if bridge_w > 1:
        bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_w, 1))
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, bridge_kernel, iterations=1)

    if intensity >= 0.28:
        smear_w = 1 + int(round(3 * intensity))
        ink = cv2.dilate(
            ink,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(2, smear_w), 1)),
            iterations=1,
        )

    if intensity >= 0.48:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        ink = cv2.dilate(ink, kernel, iterations=1)

    if intensity >= 0.34:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # Chip only edge pixels, so the result has missing/blocky contour bites
    # without turning into random scanner speckle.
    if rng.random() < 0.18 + 0.28 * intensity:
        edge = cv2.morphologyEx(ink, cv2.MORPH_GRADIENT, np.ones((2, 2), np.uint8)) > 0
        chip_noise = numpy_rng(rng).random(ink.shape) < rng.uniform(0.010, 0.035 + 0.025 * intensity)
        ink[np.logical_and(edge, chip_noise)] = 0

    up = cv2.resize(ink, (w, h), interpolation=cv2.INTER_NEAREST)
    out_gray = np.where(up > 0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_gray, cv2.COLOR_GRAY2RGB)


def safe_perspective_warp(img: np.ndarray, rng: random.Random, cfg: DegradeConfig) -> np.ndarray:
    h, w = img.shape[:2]
    pad = max(cfg.geometry_pad_px, int(max(w, h) * cfg.max_perspective_shift_frac * 2) + 6)
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    ph, pw = padded.shape[:2]
    shift = max(1, int(min(w, h) * cfg.max_perspective_shift_frac))
    src = np.float32([[pad, pad], [pad + w, pad], [pad + w, pad + h], [pad, pad + h]])
    dst = src + np.float32([[rng.randint(-shift, shift), rng.randint(-shift, shift)] for _ in range(4)])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(padded, matrix, (pw, ph), borderValue=(255, 255, 255))


def safe_rotate(pil_img: Image.Image, rng: random.Random, cfg: DegradeConfig) -> Image.Image:
    angle = rng.uniform(-cfg.max_rotation_degrees, cfg.max_rotation_degrees)
    if abs(angle) < 0.05:
        return pil_img
    pad = max(cfg.geometry_pad_px, 8)
    padded = Image.new("RGB", (pil_img.width + pad * 2, pil_img.height + pad * 2), "white")
    padded.paste(pil_img, (pad, pad))
    return padded.rotate(angle, expand=True, fillcolor="white")


def morph_ink(img: np.ndarray, rng: random.Random, mode: str) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ink = gray < 210
    mask = (ink.astype(np.uint8) * 255)
    kernel = np.ones((2, 2), np.uint8)
    if mode == "thicken":
        mask2 = cv2.dilate(mask, kernel, iterations=1)
    elif mode == "thin":
        mask2 = cv2.erode(mask, kernel, iterations=1)
    else:
        mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    out = img.copy()
    gained = mask2 > mask
    lost = mask > mask2
    out[gained] = np.minimum(out[gained], rng.randint(0, 45))
    out[lost] = np.maximum(out[lost], rng.randint(225, 255))
    return out


def apply_scan_degradation(pil_image: Image.Image, cfg: DegradeConfig, rng: random.Random) -> Image.Image:
    img = np.array(pil_image.convert("RGB"))
    profile = cfg.profile

    if profile == "browser_clean":
        return Image.fromarray(img).convert("RGB")

    if profile in {"mixed", "book_scan", "ugly_scan"}:
        profile = rng.choices(
            ["digital_clean", "raster_pdf", "book_scan", "thick_scan", "thin_scan", "overcooked_scan", "photocopy_bleed", "binary_lowres_bloat", "faded_scan"],
            weights=[10, 15, 27, 13, 9, 8, 6, 8, 4],
            k=1,
        )[0]

    if profile == "digital_clean":
        if rng.random() < 0.15:
            img = add_paper_background(img, rng, 0.25)
    elif profile == "raster_pdf":
        scale = rng.choice([0.82, 0.9, 1.15])
        if scale != 1.0:
            h, w = img.shape[:2]
            small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        img = add_paper_background(img, rng, 0.35)
        if rng.random() < 0.25:
            img = cv2.GaussianBlur(img, (3, 3), 0)
    elif profile == "book_scan":
        img = add_paper_background(img, rng, 0.9)
        if rng.random() < 0.35:
            img = morph_ink(img, rng, rng.choice(["thicken", "thin", "close"]))
        if rng.random() < 0.35:
            img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.1, 0.5))
    elif profile == "thick_scan":
        img = morph_ink(img, rng, "thicken")
        img = add_paper_background(img, rng, 0.75)
        if rng.random() < 0.30:
            img = cv2.GaussianBlur(img, (3, 3), 0)
    elif profile == "thin_scan":
        img = morph_ink(img, rng, "thin")
        img = add_paper_background(img, rng, 0.7)
        img = ImageEnhance.Contrast(Image.fromarray(img)).enhance(rng.uniform(0.72, 0.92))
        img = np.array(img)
    elif profile == "overcooked_scan":
        img = morph_ink(img, rng, "close")
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, rng.randint(4, 13))
        img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        img = add_paper_background(img, rng, 0.5)
    elif profile == "photocopy_bleed":
        intensity = rng.uniform(cfg.photocopy_bleed_intensity_min, cfg.photocopy_bleed_intensity_max)
        img = add_photocopy_bleed_pixelation(img, rng, intensity)
        if rng.random() < 0.50:
            img = add_paper_background(img, rng, 0.25)
    elif profile == "binary_lowres_bloat":
        intensity = rng.uniform(cfg.binary_bloat_intensity_min, cfg.binary_bloat_intensity_max)
        img = add_binary_lowres_bloat(img, rng, intensity)
    elif profile == "faded_scan":
        img = add_paper_background(img, rng, 1.0)
        pil = Image.fromarray(img)
        pil = ImageEnhance.Contrast(pil).enhance(rng.uniform(0.55, 0.78))
        pil = ImageEnhance.Brightness(pil).enhance(rng.uniform(1.03, 1.16))
        img = np.array(pil)
    else:
        raise ValueError(f"Unknown degradation profile: {cfg.profile}")

    if profile != "binary_lowres_bloat" and rng.random() < cfg.extra_noise_prob:
        img = add_full_image_noise(img, rng)
    if profile != "binary_lowres_bloat" and rng.random() < cfg.scanner_band_prob:
        img = add_scanner_band(img, rng)
    if profile != "binary_lowres_bloat" and rng.random() < cfg.edge_shadow_prob:
        img = add_edge_shadow(img, rng)
    if profile != "binary_lowres_bloat" and rng.random() < cfg.stain_prob:
        img = add_stains(img, rng)
    if rng.random() < cfg.nearby_rule_prob:
        img = add_nearby_rule_fragment(img, rng)
    if profile != "binary_lowres_bloat" and rng.random() < cfg.perspective_warp_prob:
        img = safe_perspective_warp(img, rng, cfg)

    pil = Image.fromarray(img)
    if rng.random() < 0.15 and profile not in {"digital_clean", "binary_lowres_bloat"}:
        pil = safe_rotate(pil, rng, cfg)
    if rng.random() < 0.18 and profile in {"book_scan", "ugly_scan", "overcooked_scan", "photocopy_bleed"}:
        pil = pil.filter(ImageFilter.UnsharpMask(radius=1.0, percent=rng.randint(110, 180), threshold=2))
    if profile != "binary_lowres_bloat" and rng.random() < cfg.heavy_blur_prob:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.45, 1.15)))

    buf = io.BytesIO()
    if rng.random() < cfg.low_quality_jpeg_prob:
        quality = rng.randint(cfg.low_quality_jpeg_min, cfg.low_quality_jpeg_max)
    else:
        quality = rng.randint(cfg.jpeg_quality_min, cfg.jpeg_quality_max)
    pil.save(buf, format="JPEG", quality=quality, optimize=False)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class Bench:
    def __init__(self, every_pages: int):
        self.every_pages = max(1, every_pages)
        self.lock = threading.Lock()
        self.explained = False
        self.reset()

    def reset(self) -> None:
        self.t0 = time.perf_counter()
        self.pages = 0
        self.lines = 0
        self.bytes = 0
        self.files = 0
        self.rejected_width = 0
        self.times = {
            "html_tmp": 0.0,
            "render": 0.0,
            "extract": 0.0,
            "screenshot": 0.0,
            "decode": 0.0,
            "crop": 0.0,
            "degrade": 0.0,
            "jpeg": 0.0,
            "write": 0.0,
            "manifest": 0.0,
            "sheet_wall": 0.0,
        }

    def add(self, metrics: dict, lines: int, files: int, bytes_written: int) -> None:
        with self.lock:
            self.pages += 1
            self.lines += lines
            self.files += files
            self.bytes += bytes_written
            self.rejected_width += int(metrics.get("rejected_width", 0))
            for k, v in metrics.items():
                if k in self.times:
                    self.times[k] += float(v)
            if self.pages >= self.every_pages:
                self.print_and_reset()

    def print_and_reset(self) -> None:
        elapsed = max(0.001, time.perf_counter() - self.t0)
        page_rate = self.pages / elapsed
        line_rate = self.lines / elapsed
        avg_lines = self.lines / max(1, self.pages)
        avg = {k: (v / max(1, self.pages)) * 1000.0 for k, v in self.times.items()}
        if not self.explained:
            print(
                "[bench_legend] *_ms values are average milliseconds per Chrome sheet. "
                "write_html_ms=write temporary HTML; chrome_screenshot_ms=Chrome writes the full sheet PNG; "
                "open_screenshot_ms=open/read PNG after Windows releases it; crop_lines_ms=crop sheet into line images; "
                "degrade_lines_ms=apply scan/noise effects; encode_jpeg_ms=JPEG encode line crops; "
                "save_images_ms=write JPEGs/manifests/metadata; sheet_total_ms=whole sheet worker time.",
                flush=True,
            )
            self.explained = True
        print(
            "[bench] "
            f"pages={self.pages} lines={self.lines} avg_lines_per_page={avg_lines:.1f} "
            f"page_rate={page_rate:.2f}/s line_rate={line_rate:.1f}/s "
            f"files={self.files} rejected_width={self.rejected_width} bytes={self.bytes / (1024 * 1024):.2f}MiB "
            f"write_html_ms={avg['html_tmp']:.1f} chrome_screenshot_ms={avg['screenshot']:.1f} "
            f"open_screenshot_ms={avg['decode']:.1f} crop_lines_ms={avg['crop']:.1f} "
            f"degrade_lines_ms={avg['degrade']:.1f} encode_jpeg_ms={avg['jpeg']:.1f} "
            f"save_images_ms={avg['write']:.1f} sheet_total_ms={avg['sheet_wall']:.1f}",
            flush=True,
        )
        self.reset()


def count_existing_manifest_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def existing_metadata_state(output_dir: Path) -> tuple[int, int]:
    metadata_path = output_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return 0, -1
    rows = 0
    max_line_id = -1
    with metadata_path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows += 1
            try:
                max_line_id = max(max_line_id, int(data.get("line_id", -1)))
            except (TypeError, ValueError):
                pass
    return rows, max_line_id


def managed_output_row_state(output_dir: Path) -> tuple[int, int, int]:
    metadata_rows, max_line_id = existing_metadata_state(output_dir)
    manifest_rows = count_existing_manifest_rows(output_dir / "train_list.txt") + count_existing_manifest_rows(output_dir / "val_list.txt")
    return metadata_rows, manifest_rows, max_line_id


def clear_managed_output(output_dir: Path) -> None:
    for name in ["images", "_tmp_html"]:
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path)
    for name in ["train_list.txt", "val_list.txt", "metadata.jsonl", "generator_config.json"]:
        path = output_dir / name
        if path.exists():
            path.unlink()


class ManifestWriter:
    def __init__(
        self,
        output_dir: Path,
        val_ratio: float,
        val_count: int | None,
        writer_workers: int,
        seed: int,
        max_lines: int,
        resume: bool,
    ):
        self.output_dir = output_dir
        self.images_dir = output_dir / "images"
        self.manifest_train = output_dir / "train_list.txt"
        self.manifest_val = output_dir / "val_list.txt"
        self.metadata_path = output_dir / "metadata.jsonl"
        self.val_ratio = val_ratio
        self.val_count = val_count
        self.max_lines = max_lines
        self.rng = random.Random(seed + 17)
        if val_count is not None:
            if max_lines <= 0:
                raise ValueError("--val-count needs --max-lines so validation can be randomly distributed across a known total image count.")
            if val_count > max_lines:
                raise ValueError(f"--val-count ({val_count}) cannot be larger than --max-lines ({max_lines}).")
            self.val_line_ids = set(self.rng.sample(range(max_lines), val_count))
            self.split_mode = "random_exact_by_line_id"
        else:
            self.val_line_ids = None
            self.split_mode = "random_ratio_per_row"
        self.lock = threading.Lock()
        self.train_rows = count_existing_manifest_rows(self.manifest_train) if resume else 0
        self.val_rows = count_existing_manifest_rows(self.manifest_val) if resume else 0
        self.total_rows = self.train_rows + self.val_rows
        self.pool = ThreadPoolExecutor(max_workers=writer_workers, thread_name_prefix="writer")
        self.pending = []

        self.images_dir.mkdir(parents=True, exist_ok=True)
        if resume:
            self.manifest_train.touch()
            self.manifest_val.touch()
            self.metadata_path.touch()
        else:
            self.manifest_train.write_text("", encoding="utf-8")
            self.manifest_val.write_text("", encoding="utf-8")
            self.metadata_path.write_text("", encoding="utf-8")

    def close(self) -> None:
        for fut in self.pending:
            fut.result()
        self.pool.shutdown(wait=True)

    def _split(self, row: dict) -> str:
        if self.val_line_ids is not None:
            return "val" if int(row["line_id"]) in self.val_line_ids else "train"
        return "val" if self.rng.random() < self.val_ratio else "train"

    def write_batch(self, rows: list[dict]) -> tuple[int, int, int, float]:
        t0 = time.perf_counter()
        files = 0
        bytes_written = 0
        train_lines = []
        val_lines = []
        metadata_lines = []
        futures = []

        with self.lock:
            for row in rows:
                split = self._split(row)
                if split == "val":
                    self.val_rows += 1
                else:
                    self.train_rows += 1
                self.total_rows += 1
                rel = f"images/{row['filename']}"
                line = f"{rel}\t{row['text']}\n"
                if split == "val":
                    val_lines.append(line)
                else:
                    train_lines.append(line)
                meta = {k: v for k, v in row.items() if k != "image_bytes"}
                meta["split"] = split
                metadata_lines.append(json.dumps(meta, ensure_ascii=False) + "\n")
                out_path = self.images_dir / row["filename"]
                payload = row["image_bytes"]
                bytes_written += len(payload)
                files += 1
                futures.append(self.pool.submit(out_path.write_bytes, payload))

            if train_lines:
                with self.manifest_train.open("a", encoding="utf-8") as f:
                    f.writelines(train_lines)
            if val_lines:
                with self.manifest_val.open("a", encoding="utf-8") as f:
                    f.writelines(val_lines)
            if metadata_lines:
                with self.metadata_path.open("a", encoding="utf-8") as f:
                    f.writelines(metadata_lines)

        for fut in futures:
            fut.result()
        return files, bytes_written, len(rows), time.perf_counter() - t0


STYLE_TOKENS = {
    "regular",
    "normal",
    "bold",
    "bd",
    "italic",
    "it",
    "oblique",
    "light",
    "lt",
    "medium",
    "med",
    "semibold",
    "semi",
    "demibold",
    "black",
    "heavy",
    "thin",
    "extra",
    "extrabold",
    "extralight",
    "condensed",
    "narrow",
}

VENDOR_TOKENS = {
    "font",
    "fonts",
    "unicode",
    "kurdish",
    "kurdfonts",
    "collection",
    "sorani",
    "central",
    "main",
    "set",
    "abd",
    "xb",
    "k",
}


def font_identity(path: Path) -> str:
    stat = path.stat()
    h = hashlib.sha1()
    h.update(str(stat.st_size).encode("ascii"))
    with path.open("rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()


def font_family_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parts = list(rel.with_suffix("").parts)
    tokens = []
    for part in parts:
        for token in re.split(r"[^0-9A-Za-z]+", part.lower()):
            if token and token not in STYLE_TOKENS and token not in VENDOR_TOKENS:
                tokens.append(token)
    stem_tokens = [
        token
        for token in re.split(r"[^0-9A-Za-z]+", path.stem.lower())
        if token and token not in STYLE_TOKENS and token not in VENDOR_TOKENS
    ]
    if stem_tokens:
        key = "_".join(stem_tokens)
    elif tokens:
        key = "_".join(tokens[-2:])
    else:
        key = re.sub(r"[^0-9A-Za-z_-]+", "_", path.stem.lower())
    return key.strip("_") or "font"


def windows_font_dirs() -> list[Path]:
    candidates = [
        Path("C:/Windows/Fonts"),
        Path("/mnt/c/Windows/Fonts"),
    ]
    return [p for p in candidates if p.exists()]


def load_fonts(
    fonts_dir: Path,
    extra_font_dirs: list[Path],
    font_extensions: set[str],
    dedupe_font_files: bool,
    group_font_families: bool,
    font_include_regex: str | None,
    font_exclude_regex: str | None,
) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    seen_identities = set()
    include_re = re.compile(font_include_regex, re.IGNORECASE) if font_include_regex else None
    exclude_re = re.compile(font_exclude_regex, re.IGNORECASE) if font_exclude_regex else None
    roots = [fonts_dir] + extra_font_dirs
    raw_files = 0
    accepted_files = 0
    duplicate_files = 0
    include_skipped = 0
    exclude_skipped = 0
    for font_root in roots:
        if not font_root.exists():
            print(f"[fonts] missing skipped dir={font_root}", flush=True)
            continue
        for root, _, files in os.walk(font_root):
            root_path = Path(root)
            for file in files:
                raw_files += 1
                ext = Path(file).suffix.lower()
                if ext not in font_extensions:
                    continue
                font_path = (root_path / file).resolve()
                if dedupe_font_files:
                    try:
                        identity = font_identity(font_path)
                    except OSError:
                        continue
                    if identity in seen_identities:
                        duplicate_files += 1
                        continue
                    seen_identities.add(identity)
                accepted_files += 1
                if group_font_families:
                    key = font_family_key(font_path, font_root.resolve())
                else:
                    rel = font_path.relative_to(font_root.resolve())
                    key = re.sub(r"[^0-9A-Za-z_-]+", "_", str(rel.with_suffix(""))).strip("_").lower()
                    if not key:
                        key = f"font_{len(families):06d}"
                font_match_text = f"{key} {font_path}".replace("\\", "/")
                if include_re and not include_re.search(font_match_text):
                    include_skipped += 1
                    continue
                if exclude_re and exclude_re.search(font_match_text):
                    exclude_skipped += 1
                    continue
                original_key = key
                suffix = 1
                while key in families and not group_font_families:
                    key = f"{original_key}_{suffix:04d}"
                    suffix += 1
                families.setdefault(key, []).append(str(font_path).replace("\\", "/"))
    if not families:
        raise FileNotFoundError(f"No fonts found under {fonts_dir} with extensions={sorted(font_extensions)}")
    print(
        f"[fonts] raw_files={raw_files} accepted_files={accepted_files} "
        f"duplicate_files_skipped={duplicate_files} families={len(families)} "
        f"include_regex_skipped={include_skipped} exclude_regex_skipped={exclude_skipped} "
        f"extensions={','.join(sorted(font_extensions))}",
        flush=True,
    )
    return families




def choose_unit_word_count(rng: random.Random, min_words: int, max_words: int, mode_words: int) -> int:
    return max(min_words, min(max_words, int(round(rng.triangular(min_words, max_words, mode_words)))))


def target_unit_settings(args) -> tuple[int, int, int, int, int]:
    if args.target_unit == "line":
        return (
            args.line_words_min,
            args.line_words_max,
            args.line_words_mode,
            args.line_chars_min,
            args.line_chars_max,
        )
    if args.target_unit == "word":
        return (
            args.word_words_min,
            args.word_words_max,
            args.word_words_mode,
            args.word_chars_min,
            args.word_chars_max,
        )
    if args.target_unit == "short_phrase":
        return (
            args.short_phrase_words_min,
            args.short_phrase_words_max,
            args.short_phrase_words_mode,
            args.short_phrase_chars_min,
            args.short_phrase_chars_max,
        )
    if args.target_unit == "phrase48":
        return (
            args.phrase48_words_min,
            args.phrase48_words_max,
            args.phrase48_words_mode,
            args.phrase48_chars_min,
            args.phrase48_chars_max,
        )
    raise ValueError(f"unknown target unit: {args.target_unit}")


def has_latin_letter(text: str) -> bool:
    return any(("A" <= ch <= "Z") or ("a" <= ch <= "z") or ch in "ÇĞİÖŞÜçğıöşü" for ch in text)


def has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def is_arabic_script_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x0600 <= code <= 0x06FF
        or 0x0750 <= code <= 0x077F
        or 0x08A0 <= code <= 0x08FF
        or 0xFB50 <= code <= 0xFDFF
        or 0xFE70 <= code <= 0xFEFF
    )


def passes_script_filter(text: str, script_filter: str) -> bool:
    has_arabic = any(is_arabic_script_char(ch) for ch in text)
    if script_filter == "any":
        return True
    if script_filter == "arabic_no_latin":
        return has_arabic and not has_latin_letter(text)
    if script_filter == "arabic_no_latin_digit":
        return has_arabic and not has_latin_letter(text) and not has_digit(text)
    if script_filter == "arabic_letters_marks_space":
        return has_arabic and all(ch.isspace() or is_arabic_script_char(ch) for ch in text)
    raise ValueError(f"unknown script filter: {script_filter}")


def text_units(text: str, rng: random.Random, args) -> list[str]:
    min_words, max_words, mode_words, min_chars, max_chars = target_unit_settings(args)
    units = []
    for paragraph in re.split(r"[\n؟?!؛;。]+", text):
        paragraph = paragraph.strip()
        if len(paragraph) < 4:
            continue
        words = paragraph.split()
        start = 0
        while start < len(words):
            length = choose_unit_word_count(rng, min_words, max_words, mode_words)
            end = min(len(words), start + length)
            line = " ".join(words[start:end]).strip()
            if min_chars <= len(line) <= max_chars and passes_script_filter(line, args.script_filter):
                units.append(line)
            start = max(end, start + 1)
    return units


def read_label_source_manifest(args) -> list[str]:
    manifest_path = Path(args.label_source_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"--label-source-manifest not found: {manifest_path}")

    labels: list[str] = []
    raw_rows = 0
    empty_rows = 0
    filtered_rows = 0
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            raw_rows += 1
            line = line.rstrip("\n")
            if not line:
                empty_rows += 1
                continue
            if "\t" in line:
                _, label = line.split("\t", 1)
            else:
                label = line
            label = normalize_ocr_label_text(
                label,
                profile=args.text_cleanup_profile,
                strip_arabic_marks=args.strip_arabic_marks,
                normalize_arabic_yeh_nonfinal=args.normalize_arabic_yeh_nonfinal,
                normalize_zwnj=args.normalize_zwnj,
            )
            if not label:
                empty_rows += 1
                continue
            if not passes_script_filter(label, args.script_filter):
                filtered_rows += 1
                continue
            if args.label_source_dedupe and label in seen:
                filtered_rows += 1
                continue
            seen.add(label)
            labels.append(label)

    if args.label_source_shuffle:
        shuffle_seed = args.label_source_shuffle_seed if args.label_source_shuffle_seed is not None else args.seed + 31337
        rng = random.Random(shuffle_seed)
        rng.shuffle(labels)
    else:
        shuffle_seed = None

    print(
        "[label_source] "
        f"manifest={manifest_path} raw_rows={raw_rows} accepted_labels={len(labels)} "
        f"empty_rows={empty_rows} filtered_or_duplicate_rows={filtered_rows} "
        f"repeat={args.label_source_repeat} shuffle={args.label_source_shuffle} "
        f"shuffle_seed={shuffle_seed} "
        f"dedupe={args.label_source_dedupe}",
        flush=True,
    )
    if not labels:
        raise RuntimeError(f"--label-source-manifest produced zero usable labels: {manifest_path}")
    return labels


def make_fixed_label_tasks(args, families: dict[str, list[str]]) -> Iterable[dict]:
    rng = random.Random(args.seed)
    labels = read_label_source_manifest(args)
    repeat = max(1, int(args.label_source_repeat))
    resume_start_line_id = int(getattr(args, "resume_start_line_id", 0) or 0)
    line_id = 0
    for label_index, text_line in enumerate(labels):
        for repeat_index in range(repeat):
            font_family = rng.choice(list(families.keys()))
            font_path = rng.choice(families[font_family])
            if line_id < resume_start_line_id:
                line_id += 1
                if args.max_lines and line_id >= args.max_lines:
                    return
                continue
            yield {
                "line_id": line_id,
                "text": text_line,
                "font_family": font_family,
                "font_path": font_path,
                "source_path": str(Path(args.label_source_manifest).resolve()),
                "source_type": "label_source_manifest",
                "label_source_index": label_index,
                "label_repeat_index": repeat_index,
            }
            line_id += 1
            if args.max_lines and line_id >= args.max_lines:
                return


def pil_has_raqm() -> bool:
    try:
        return bool(features.check("raqm"))
    except Exception:
        return False


def pil_text_layout_kwargs() -> dict:
    if pil_has_raqm():
        return {"direction": "rtl", "language": "ku"}
    return {}


def shape_text_for_pil(text: str) -> str:
    if pil_has_raqm():
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


def fit_font(font_path: str, text: str, target_height: int, max_width: int, rng: random.Random) -> tuple[ImageFont.FreeTypeFont, str, tuple[int, int, int, int]]:
    shaped = shape_text_for_pil(text)
    layout_kwargs = pil_text_layout_kwargs()
    size = rng.randint(22, 42)
    for _ in range(20):
        try:
            font = ImageFont.truetype(font_path, size=size)
        except OSError as exc:
            raise RuntimeError(f"Pillow fallback renderer cannot load font_path={font_path!r}: {exc}") from exc
        bbox = ImageDraw.Draw(Image.new("RGB", (10, 10))).textbbox((0, 0), shaped, font=font, **layout_kwargs)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= max_width and h <= target_height:
            return font, shaped, bbox
        size = max(10, size - 2)
    try:
        font = ImageFont.truetype(font_path, size=max(10, size))
    except OSError as exc:
        raise RuntimeError(f"Pillow fallback renderer cannot load font_path={font_path!r}: {exc}") from exc
    bbox = ImageDraw.Draw(Image.new("RGB", (10, 10))).textbbox((0, 0), shaped, font=font, **layout_kwargs)
    return font, shaped, bbox


def render_line_image(text: str, font_path: str, rng: random.Random) -> Image.Image:
    max_width = rng.choice([768, 960, 1152, 1344, 1536])
    target_height = rng.choice([42, 48, 56, 64])
    font, shaped, bbox = fit_font(font_path, text, target_height - 12, max_width - 32, rng)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x = rng.randint(8, 20)
    pad_y = rng.randint(5, 11)
    width = min(max_width, max(48, text_w + pad_x * 2))
    height = max(22, text_h + pad_y * 2)
    paper = rng.choice([(255, 255, 255), (251, 250, 246), (247, 244, 235), (238, 238, 232)])
    img = Image.new("RGB", (width, height), paper)
    draw = ImageDraw.Draw(img)
    x = max(0, width - pad_x - text_w)
    y = pad_y - bbox[1]
    color = rng.choice([(0, 0, 0), (12, 12, 12), (25, 25, 25), (38, 38, 38)])
    draw.text((x, y), shaped, font=font, fill=color, **pil_text_layout_kwargs())

    if rng.random() < 0.10:
        yline = rng.choice([1, height - 2, rng.randint(0, max(0, height - 1))])
        draw.line((0, yline, width, yline), fill=rng.choice([(0, 0, 0), (80, 80, 80), (160, 160, 160)]), width=rng.choice([1, 1, 2]))
    if rng.random() < 0.04:
        xline = rng.choice([1, width - 2])
        draw.line((xline, 0, xline, height), fill=(120, 120, 120), width=1)
    return img


def find_browser(explicit: str | None) -> str:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Browser executable not found: {explicit}")
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        shutil.which("chrome"),
        shutil.which("msedge"),
        shutil.which("chromium"),
        shutil.which("google-chrome"),
    ]
    for item in candidates:
        if item and Path(item).exists():
            return str(item)
    raise FileNotFoundError("Could not find Chrome or Edge. Pass --browser-exe explicitly.")


def build_browser_sheet_html(lines: list[dict], page_id: int, rng: random.Random, args) -> tuple[str, list[dict], dict]:
    sheet_w = rng.choice([1240, 1536, 1728])
    row_h = rng.choice([58, 64, 72])
    pad_x = rng.randint(16, 28)
    pad_y = rng.randint(8, 14)
    gap = rng.randint(6, 10)
    max_height = int(getattr(args, "browser_max_screenshot_height", 0) or 0)
    compacted_for_height = False
    if max_height > 0 and lines:
        min_row_h = 42
        min_gap = 4
        while len(lines) * (row_h + gap) + gap > max_height and (row_h > min_row_h or gap > min_gap):
            compacted_for_height = True
            if gap > min_gap:
                gap -= 1
            elif row_h > min_row_h:
                row_h -= 2
    sheet_h = max(64, len(lines) * (row_h + gap) + gap)
    paper = rng.choice(["#ffffff", "#fbfaf6", "#f7f4ea", "#f1eee5", "#eeeeea"])
    color = rng.choice(["#000000", "#111111", "#181818", "#222222"])
    items = []
    meta = []
    for i, line in enumerate(lines):
        top = gap + i * (row_h + gap)
        max_font_size = max(24, min(40, row_h - 8))
        font_size = rng.randint(24, max_font_size)
        font_path = line["font_path"].replace("\\", "/")
        family = f"F{i}"
        css_weight = rng.choice(["400", "400", "500", "600", "700"])
        css_line_height = rng.randint(max(font_size + 6, row_h - 8), row_h)
        opacity = rng.uniform(0.72, 0.96) if rng.random() < args.text_opacity_prob else 1.0
        word_spacing = (
            rng.uniform(args.word_spacing_min_px, args.word_spacing_max_px)
            if rng.random() < args.word_spacing_prob
            else 0.0
        )
        letter_spacing = (
            rng.uniform(args.letter_spacing_min_px, args.letter_spacing_max_px)
            if rng.random() < args.letter_spacing_prob
            else 0.0
        )
        items.append(
            f"""@font-face {{ font-family:'{family}'; src:url('file:///{font_path}'); }}
.line-{i} {{
  position:absolute; left:{pad_x}px; top:{top}px; width:{sheet_w - pad_x * 2}px; height:{row_h}px;
  overflow:hidden; direction:rtl; unicode-bidi:plaintext; text-align:right;
  font-family:'{family}', serif; font-size:{font_size}px; font-weight:{css_weight};
  line-height:{css_line_height}px; white-space:nowrap; color:{color}; opacity:{opacity:.3f};
  word-spacing:{word_spacing:.2f}px; letter-spacing:{letter_spacing:.2f}px;
}}"""
        )
        items.append(f"<div class='line-{i}'>{html.escape(line['text'])}</div>")
        meta.append({
            **line,
            "sheet_page_id": page_id,
            "sheet_line_index": i,
            "dom_class": f"line-{i}",
            "crop_box": [0, max(0, top - min(pad_y, max(0, gap // 2))), sheet_w, min(sheet_h, top + row_h + min(pad_y, max(0, gap // 2)))],
            "font_size": font_size,
            "css_weight": css_weight,
            "css_line_height": css_line_height,
            "css_opacity": round(opacity, 3),
            "css_word_spacing_px": round(word_spacing, 2),
            "css_letter_spacing_px": round(letter_spacing, 2),
        })
    doc = f"""<!doctype html>
<html lang="ku" dir="rtl">
<head><meta charset="utf-8">
<style>
html, body {{ margin:0; padding:0; width:{sheet_w}px; height:{sheet_h}px; overflow:hidden; background:{paper}; }}
body {{ position:relative; }}
{os.linesep.join(x for x in items if x.startswith('@font-face') or x.startswith('.line-'))}
</style></head>
<body>
{os.linesep.join(x for x in items if x.startswith('<div'))}
<script>
function fitOcrLine(el) {{
  const minFontPx = 14;
  let size = parseFloat(getComputedStyle(el).fontSize);
  while ((el.scrollWidth > el.clientWidth + 1 || el.scrollHeight > el.clientHeight + 1) && size > minFontPx) {{
    size -= 1;
    el.style.fontSize = size + 'px';
    el.style.lineHeight = Math.max(size + 4, Math.min(el.clientHeight, size * 1.35)) + 'px';
  }}
}}
Array.from(document.querySelectorAll('div[class^="line-"]')).forEach(fitOcrLine);
</script>
</body></html>"""
    return doc, meta, {
        "sheet_width": sheet_w,
        "sheet_height": sheet_h,
        "row_height": row_h,
        "gap": gap,
        "compacted_for_height": compacted_for_height,
    }


def screenshot_with_browser(
    browser_exe: str,
    html_path: Path,
    out_png: Path,
    profile_dir: Path,
    width: int,
    height: int,
    timeout: int,
    retries: int,
    retry_delay: float,
    settle_seconds: float,
    settle_poll_seconds: float,
    verbose_log: bool,
) -> None:
    def native_path(path: Path) -> str:
        return os.path.abspath(str(path))

    def wait_for_screenshot(wait_seconds: float) -> tuple[bool, int]:
        deadline = time.perf_counter() + max(0.0, wait_seconds)
        last_size = 0
        while True:
            exists = out_png.exists()
            size = out_png.stat().st_size if exists else 0
            if exists and size > 0:
                try:
                    with out_png.open("rb") as f:
                        f.read(16)
                    return True, size
                except OSError:
                    pass
            last_size = size
            if time.perf_counter() >= deadline:
                return exists, last_size
            time.sleep(max(0.01, settle_poll_seconds))

    attempts = max(0, retries) + 1
    failures = []
    for attempt in range(1, attempts + 1):
        attempt_profile_dir = profile_dir / f"attempt_{attempt:02d}"
        if attempt_profile_dir.exists():
            shutil.rmtree(attempt_profile_dir, ignore_errors=True)
        attempt_profile_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = attempt_profile_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        log_path = attempt_profile_dir / "chrome_stdout_stderr.log"
        cmd = [
            browser_exe,
            "--headless=new",
            "--enable-logging=stderr",
            "--disable-gpu",
            "--no-first-run",
            "--noerrdialogs",
            "--disable-breakpad",
            "--disable-crash-reporter",
            "--disable-background-networking",
            "--disable-extensions",
            "--disable-component-update",
            "--hide-scrollbars",
            f"--user-data-dir={native_path(attempt_profile_dir)}",
            f"--disk-cache-dir={native_path(cache_dir)}",
            "--allow-file-access-from-files",
            f"--window-size={width},{height}",
            f"--screenshot={native_path(out_png)}",
            "file:///" + str(html_path).replace("\\", "/"),
        ]
        if verbose_log:
            cmd.insert(3, "--v=1")
        if out_png.exists():
            out_png.unlink()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            chrome_output, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_process_tree(proc)
            try:
                chrome_output, _ = proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                chrome_output = ""
            log_path.write_text(chrome_output or "", encoding="utf-8", errors="replace")
            failures.append(
                f"attempt={attempt}/{attempts} timeout_seconds={timeout} "
                f"profile_dir={attempt_profile_dir} chrome_log={log_path} partial_output={(chrome_output or '')[-1000:]!r}"
            )
            if attempt < attempts and retry_delay > 0:
                time.sleep(retry_delay)
            continue
        log_path.write_text(chrome_output or "", encoding="utf-8", errors="replace")
        settle_wait = max(settle_seconds, float(timeout)) if proc.returncode == 0 else settle_seconds
        exists, size = wait_for_screenshot(settle_wait)
        if proc.returncode == 0 and exists and size > 0:
            return
        failures.append(
            f"attempt={attempt}/{attempts} rc={proc.returncode} "
            f"profile_dir={attempt_profile_dir} screenshot_exists={exists} screenshot_bytes={size} "
            f"chrome_log={log_path} chrome_output_tail={(chrome_output or '')[-1000:]!r}"
        )
        if attempt < attempts and retry_delay > 0:
            time.sleep(retry_delay)
    raise RuntimeError(
        "browser screenshot failed after retries: "
        f"html_path={html_path} out_png={out_png} profile_dir={profile_dir} "
        f"width={width} height={height} attempts={attempts}; "
        + " | ".join(failures)
    )


def native_path(path: Path) -> str:
    return os.path.abspath(str(path))




def terminate_process_tree(proc: subprocess.Popen, wait_seconds: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            proc.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            pass
        return
    proc.terminate()
    try:
        proc.wait(timeout=wait_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=wait_seconds)














def open_image_after_browser(path: Path, settle_seconds: float, poll_seconds: float) -> Image.Image:
    deadline = time.perf_counter() + max(0.0, settle_seconds)
    last_error = None
    while True:
        try:
            with Image.open(path) as img:
                img.load()
                return img.convert("RGB")
        except (OSError, PermissionError) as exc:
            last_error = exc
            if time.perf_counter() >= deadline:
                raise RuntimeError(f"Could not open Chrome screenshot after waiting: path={path} last_error={last_error}") from exc
            time.sleep(max(0.01, poll_seconds))


def trim_line_margins(img: Image.Image, pad_x: int, pad_y: int) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    bg = np.percentile(gray, 92)
    mask = gray < max(0, bg - 18)
    ys, xs = np.where(mask)
    if len(xs) < 8 or len(ys) < 8:
        return img
    left = max(0, int(xs.min()) - pad_x)
    right = min(img.width, int(xs.max()) + pad_x + 1)
    top = max(0, int(ys.min()) - pad_y)
    bottom = min(img.height, int(ys.max()) + pad_y + 1)
    if right - left < 24 or bottom - top < 12:
        return img
    return img.crop((left, top, right, bottom))


def render_browser_sheet_batch_once(
    args,
    browser: str,
    tmp_dir: Path,
    degrade_cfg: DegradeConfig,
    page_id: int,
    batch: list[dict],
) -> tuple[list[dict], dict]:
    sheet_t0 = time.perf_counter()
    rng = random.Random(args.seed + page_id * 15485863)
    html_doc, line_meta, sheet_meta = build_browser_sheet_html(batch, page_id, rng, args)
    sheet_key = f"sheet_{page_id:08d}_{batch[0]['line_id']:010d}_{batch[-1]['line_id']:010d}"
    html_path = tmp_dir / f"{sheet_key}.html"
    png_path = tmp_dir / f"{sheet_key}.png"
    profile_dir = tmp_dir / f"chrome_profile_{sheet_key}"
    max_screenshot_height = int(getattr(args, "browser_max_screenshot_height", 0) or 0)
    if max_screenshot_height > 0 and sheet_meta["sheet_height"] > max_screenshot_height:
        suggested_sheet_lines = max(1, int(args.sheet_lines * max_screenshot_height / sheet_meta["sheet_height"]))
        raise RuntimeError(
            "browser sheet is too tall for reliable Chrome screenshots: "
            f"sheet_height={sheet_meta['sheet_height']} browser_max_screenshot_height={max_screenshot_height} "
            f"sheet_lines={args.sheet_lines}. Re-run with --sheet-lines {suggested_sheet_lines} or lower, "
            "or pass --browser-max-screenshot-height 0 to disable this guard."
        )
    metrics = {
        "html_tmp": 0.0,
        "render": 0.0,
        "extract": 0.0,
        "screenshot": 0.0,
        "decode": 0.0,
        "crop": 0.0,
        "degrade": 0.0,
        "jpeg": 0.0,
        "write": 0.0,
        "manifest": 0.0,
        "sheet_wall": 0.0,
        "rejected_width": 0,
    }
    keep_debug_files = False
    try:
        t = time.perf_counter()
        html_path.write_text(html_doc, encoding="utf-8")
        metrics["html_tmp"] = time.perf_counter() - t

        t = time.perf_counter()
        try:
            screenshot_with_browser(
                browser,
                html_path,
                png_path,
                profile_dir,
                sheet_meta["sheet_width"],
                sheet_meta["sheet_height"],
                args.browser_timeout,
                args.browser_retries,
                args.browser_retry_delay,
                args.browser_screenshot_settle_seconds,
                args.browser_screenshot_settle_poll,
                args.browser_verbose_log,
            )
        except Exception:
            keep_debug_files = True
            raise
        metrics["screenshot"] = time.perf_counter() - t

        t = time.perf_counter()
        sheet_img = open_image_after_browser(
            png_path,
            args.browser_screenshot_settle_seconds,
            args.browser_screenshot_settle_poll,
        )
        metrics["decode"] = time.perf_counter() - t

        rows = []
        for item in line_meta:
            crop_box = item["crop_box"]
            line_rng = random.Random(args.seed + item["line_id"] * 7919)

            t = time.perf_counter()
            crop = sheet_img.crop(tuple(crop_box))
            metrics["crop"] += time.perf_counter() - t

            t = time.perf_counter()
            crop_pad_x = line_rng.randint(args.crop_pad_x_min, args.crop_pad_x_max)
            crop_pad_y = line_rng.randint(args.crop_pad_y_min, args.crop_pad_y_max)
            crop = trim_line_margins(crop, crop_pad_x, crop_pad_y)
            crop = apply_scan_degradation(crop, degrade_cfg, line_rng)
            metrics["degrade"] += time.perf_counter() - t
            if args.reject_crop_width_over > 0 and crop.width > args.reject_crop_width_over:
                metrics["rejected_width"] += 1
                continue

            t = time.perf_counter()
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=88, optimize=False)
            metrics["jpeg"] += time.perf_counter() - t

            rows.append({
                "filename": f"line_{item['line_id']:010d}.jpg",
                "text": item["text"],
                "line_id": item["line_id"],
                "font_family": item["font_family"],
                "font_path": item["font_path"],
                "source_path": item["source_path"],
                "source_type": item["source_type"],
                "label_source_index": item.get("label_source_index"),
                "label_repeat_index": item.get("label_repeat_index"),
                "degradation_profile": degrade_cfg.profile,
                "image_width": crop.width,
                "image_height": crop.height,
                "bbox": [0, 0, crop.width, crop.height],
                "sheet_page_id": page_id,
                "sheet_line_index": item["sheet_line_index"],
                "sheet_height": sheet_meta["sheet_height"],
                "sheet_row_height": sheet_meta["row_height"],
                "sheet_gap": sheet_meta["gap"],
                "sheet_compacted_for_height": sheet_meta.get("compacted_for_height", False),
                "font_size": item["font_size"],
                "css_weight": item["css_weight"],
                "css_line_height": item["css_line_height"],
                "css_opacity": item["css_opacity"],
                "css_word_spacing_px": item["css_word_spacing_px"],
                "css_letter_spacing_px": item["css_letter_spacing_px"],
                "crop_pad_x": crop_pad_x,
                "crop_pad_y": crop_pad_y,
                "image_bytes": buf.getvalue(),
            })

        denom = max(1, len(rows))
        metrics["crop"] /= denom
        metrics["degrade"] /= denom
        metrics["jpeg"] /= denom
        metrics["sheet_wall"] = time.perf_counter() - sheet_t0
        return rows, metrics
    finally:
        if not keep_debug_files:
            for path in (html_path, png_path):
                try:
                    path.unlink()
                except OSError:
                    pass
            try:
                shutil.rmtree(profile_dir, ignore_errors=True)
            except OSError:
                pass
        else:
            print(
                f"[browser] kept failed debug files html_path={html_path} png_path={png_path} profile_dir={profile_dir}",
                flush=True,
            )


def render_rows_with_pil_fallback(
    args,
    degrade_cfg: DegradeConfig,
    page_id: int,
    batch: list[dict],
    reason: str,
) -> tuple[list[dict], dict]:
    t0 = time.perf_counter()
    rows = []
    metrics = {
        "html_tmp": 0.0,
        "render": 0.0,
        "extract": 0.0,
        "screenshot": 0.0,
        "decode": 0.0,
        "crop": 0.0,
        "degrade": 0.0,
        "jpeg": 0.0,
        "write": 0.0,
        "manifest": 0.0,
        "sheet_wall": 0.0,
    }
    print(
        "[browser] final fallback renderer: "
        f"page_id={page_id} rows={len(batch)} line_id_range={batch[0]['line_id']}-{batch[-1]['line_id']} "
        f"reason={reason[:300]!r}",
        flush=True,
    )
    for item in batch:
        line_rng = random.Random(args.seed + item["line_id"] * 7919)

        t = time.perf_counter()
        img = render_line_image(item["text"], item["font_path"], line_rng)
        metrics["render"] += time.perf_counter() - t

        t = time.perf_counter()
        img = apply_scan_degradation(img, degrade_cfg, line_rng)
        metrics["degrade"] += time.perf_counter() - t

        t = time.perf_counter()
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88, optimize=False)
        metrics["jpeg"] += time.perf_counter() - t

        rows.append({
            "filename": f"line_{item['line_id']:010d}.jpg",
            "text": item["text"],
            "line_id": item["line_id"],
            "font_family": item["font_family"],
            "font_path": item["font_path"],
            "source_path": item["source_path"],
            "source_type": item["source_type"],
            "label_source_index": item.get("label_source_index"),
            "label_repeat_index": item.get("label_repeat_index"),
            "degradation_profile": degrade_cfg.profile,
            "image_width": img.width,
            "image_height": img.height,
            "bbox": [0, 0, img.width, img.height],
            "sheet_page_id": page_id,
            "sheet_line_index": -1,
            "sheet_height": img.height,
            "sheet_row_height": img.height,
            "sheet_gap": 0,
            "sheet_compacted_for_height": False,
            "render_fallback": "pil_line",
            "fallback_reason": reason[:500],
            "image_bytes": buf.getvalue(),
        })

    denom = max(1, len(rows))
    metrics["render"] /= denom
    metrics["degrade"] /= denom
    metrics["jpeg"] /= denom
    metrics["sheet_wall"] = time.perf_counter() - t0
    return rows, metrics


def render_browser_sheet_batch(
    args,
    browser: str,
    tmp_dir: Path,
    degrade_cfg: DegradeConfig,
    page_id: int,
    batch: list[dict],
) -> tuple[list[dict], dict]:
    def merge(left: tuple[list[dict], dict], right: tuple[list[dict], dict]) -> tuple[list[dict], dict]:
        left_rows, left_metrics = left
        right_rows, right_metrics = right
        rows = left_rows + right_rows
        total = max(1, len(rows))
        merged = {k: 0.0 for k in left_metrics}
        per_row = {"crop", "degrade", "jpeg"}
        for key in merged:
            if key in per_row:
                merged[key] = (
                    left_metrics.get(key, 0.0) * len(left_rows)
                    + right_metrics.get(key, 0.0) * len(right_rows)
                ) / total
            else:
                merged[key] = left_metrics.get(key, 0.0) + right_metrics.get(key, 0.0)
        return rows, merged

    def run_part(part_page_id: int, part_batch: list[dict], depth: int) -> tuple[list[dict], dict]:
        try:
            return render_browser_sheet_batch_once(args, browser, tmp_dir, degrade_cfg, part_page_id, part_batch)
        except RuntimeError as exc:
            message = str(exc)
            is_browser_capture_failure = (
                "browser screenshot failed" in message
                or "browser sheet is too tall" in message
            )
            if not is_browser_capture_failure or len(part_batch) <= 1:
                if is_browser_capture_failure and args.fallback_renderer == "pil":
                    # The batch cannot be split any further and Pillow is allowed,
                    # so keep the run alive instead of losing the whole sheet.
                    return render_rows_with_pil_fallback(
                        args,
                        degrade_cfg,
                        part_page_id,
                        part_batch,
                        reason=message,
                    )
                raise
            mid = len(part_batch) // 2
            left = part_batch[:mid]
            right = part_batch[mid:]
            print(
                "[browser] retrying failed Chrome sheet as smaller screenshots: "
                f"page_id={part_page_id} rows={len(part_batch)} "
                f"left_rows={len(left)} right_rows={len(right)} "
                f"line_id_range={part_batch[0]['line_id']}-{part_batch[-1]['line_id']} "
                f"reason={message[:300]!r}",
                flush=True,
            )
            left_result = run_part(part_page_id * 2 + 1, left, depth + 1)
            right_result = run_part(part_page_id * 2 + 2, right, depth + 1)
            return merge(left_result, right_result)

    return run_part(page_id, batch, 0)


def make_degrade_config(args) -> DegradeConfig:
    return DegradeConfig(
        profile=args.degradation_profile,
        jpeg_quality_min=args.jpeg_quality_min,
        jpeg_quality_max=args.jpeg_quality_max,
        perspective_warp_prob=args.perspective_warp_prob,
        max_perspective_shift_frac=args.max_perspective_shift_frac,
        geometry_pad_px=args.geometry_pad_px,
        max_rotation_degrees=args.max_rotation_degrees,
        scanner_band_prob=args.scanner_band_prob,
        edge_shadow_prob=args.edge_shadow_prob,
        stain_prob=args.stain_prob,
        extra_noise_prob=args.extra_noise_prob,
        nearby_rule_prob=args.nearby_rule_prob,
        heavy_blur_prob=args.heavy_blur_prob,
        low_quality_jpeg_prob=args.low_quality_jpeg_prob,
        low_quality_jpeg_min=args.low_quality_jpeg_min,
        low_quality_jpeg_max=args.low_quality_jpeg_max,
        photocopy_bleed_intensity_min=args.photocopy_bleed_intensity_min,
        photocopy_bleed_intensity_max=args.photocopy_bleed_intensity_max,
        binary_bloat_intensity_min=args.binary_bloat_intensity_min,
        binary_bloat_intensity_max=args.binary_bloat_intensity_max,
    )


def effective_sheet_schedule(args) -> tuple[int, int, str]:
    requested_lines = max(1, int(args.sheet_lines))
    requested_workers = max(1, int(args.sheet_workers))
    reason = "using requested sheet_lines/sheet_workers; failed sheets split inside the same worker"
    return requested_lines, requested_workers, reason


def run_browser_sheet(args, families: dict[str, list[str]], writer: ManifestWriter, bench: Bench) -> None:
    browser = find_browser(args.browser_exe)
    effective_sheet_lines, effective_sheet_workers, schedule_reason = effective_sheet_schedule(args)
    print(f"[browser] executable={browser}", flush=True)
    print("[browser] capture=chrome_cli_screenshot waits_for_png_ready=true", flush=True)
    print(
        f"[parallel] requested_sheet_workers={args.sheet_workers} effective_sheet_workers={effective_sheet_workers} "
        f"writer_workers={args.writer_workers} requested_sheet_lines={args.sheet_lines} "
        f"effective_sheet_lines={effective_sheet_lines} reason={schedule_reason}",
        flush=True,
    )
    line_iter = make_line_tasks(args, families)
    degrade_cfg = make_degrade_config(args)
    tmp_dir = Path(args.tmp_dir)
    sheet_pool = ThreadPoolExecutor(max_workers=effective_sheet_workers, thread_name_prefix="sheet")
    pending: list[tuple[int, object]] = []
    page_id = 0
    submitted = 0
    completed_sheets = 0

    def write_completed(future) -> None:
        nonlocal submitted, completed_sheets
        rows, metrics = future.result()
        if rows:
            files, bytes_written, row_count, write_time = writer.write_batch(rows)
            metrics["write"] = write_time
            bench.add(metrics, row_count, files, bytes_written)
            submitted += row_count
        completed_sheets += 1

    try:
        while True:
            batch = []
            try:
                for _ in range(effective_sheet_lines):
                    batch.append(next(line_iter))
            except StopIteration:
                pass
            if not batch:
                break

            future = sheet_pool.submit(render_browser_sheet_batch, args, browser, tmp_dir, degrade_cfg, page_id, batch)
            pending.append((page_id, future))
            page_id += 1

            while len(pending) >= effective_sheet_workers:
                _, oldest = pending.pop(0)
                write_completed(oldest)

        for _, future in pending:
            write_completed(future)
    finally:
        sheet_pool.shutdown(wait=True)

    print(f"[engine] submitted_lines={submitted} sheet_pages={completed_sheets}", flush=True)


def make_line_tasks(args, families: dict[str, list[str]]) -> Iterable[dict]:
    if args.label_source_manifest:
        yield from make_fixed_label_tasks(args, families)
        return

    rng = random.Random(args.seed)
    streamer = TextStreamer(
        args.pseudo_kurdish,
        args.max_seen_docs,
        args.text_cleanup_profile,
        args.strip_arabic_marks,
        args.normalize_arabic_yeh_nonfinal,
        args.normalize_zwnj,
        args.convert_latin_kurdish_to_arabic,
    )
    sources = source_specs(args)
    resume_start_line_id = int(getattr(args, "resume_start_line_id", 0) or 0)
    active = []
    for path, kind in sources:
        if not os.path.exists(path):
            print(f"[source] missing skipped path={path} type={kind}", flush=True)
            continue
        active.append((path, kind, streamer.stream_file(path, kind)))
    print(f"[source] active_sources={len(active)} requested_sources={len(sources)}", flush=True)
    if not active:
        # Without this the run would "succeed" with an empty dataset, which is
        # far harder to notice than a hard failure.
        listed = "\n  ".join(f"{path}  ({kind})" for path, kind in sources)
        raise FileNotFoundError(
            f"None of the {len(sources)} requested source files could be opened:\n  {listed}\n"
            "Check the paths, or drop --source and use --use-default-sources to scan "
            "--default-sources-dir."
        )
    line_id = 0
    while active:
        path, kind, gen = rng.choice(active)
        try:
            _, text = next(gen)
        except StopIteration:
            active = [x for x in active if x[2] is not gen]
            continue
        for text_line in text_units(text, rng, args):
            font_family = rng.choice(list(families.keys()))
            font_path = rng.choice(families[font_family])
            if line_id < resume_start_line_id:
                line_id += 1
                if args.max_lines and line_id >= args.max_lines:
                    return
                continue
            yield {
                "line_id": line_id,
                "text": text_line,
                "font_family": font_family,
                "font_path": font_path,
                "source_path": path,
                "source_type": kind,
            }
            line_id += 1
            if args.max_lines and line_id >= args.max_lines:
                return












def source_specs(args) -> list[tuple[str, str]]:
    specs = []
    raw = args.source or []
    if args.use_default_sources:
        discovered = discover_default_sources(Path(args.default_sources_dir))
        print(
            f"[source] default_sources_dir={args.default_sources_dir} discovered={len(discovered)}",
            flush=True,
        )
        raw = raw + [f"{path}:{kind}" for path, kind in discovered]
    for item in raw:
        if ":" not in item:
            raise ValueError(f"--source must be PATH:TYPE, got {item!r}")
        path, kind = item.rsplit(":", 1)
        kind = kind.strip().lower()
        if kind not in SOURCE_TYPES:
            raise ValueError(f"--source type must be one of {', '.join(SOURCE_TYPES)}, got {kind!r}")
        specs.append((path, kind))
    if not specs and not args.label_source_manifest:
        raise ValueError(
            "No text sources configured. Pass --source PATH:TYPE at least once, "
            "or pass --label-source-manifest FILE to render an existing label list."
        )
    return specs


def print_source_availability(specs: list[tuple[str, str]]) -> int:
    """Report which source files are readable. Returns the number missing."""
    present = []
    missing = []
    for path, kind in specs:
        (present if os.path.exists(path) else missing).append((path, kind))
    print(f"[source] requested={len(specs)} present={len(present)} missing={len(missing)}", flush=True)
    for path, kind in missing[:20]:
        print(f"[source] missing path={path} type={kind}", flush=True)
    for path, kind in present[:20]:
        print(f"[source] present path={path} type={kind}", flush=True)
    if missing and not present:
        print("[source] no requested source file is readable; generation would produce nothing.", flush=True)
    return len(missing)


def download_with_progress(url: str, out_path: Path, chunk_size: int = 8 * 1024 * 1024) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[download] exists; keeping {out_path} size={out_path.stat().st_size:,}", flush=True)
        return out_path
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    print(f"[download] url={url}", flush=True)
    print(f"[download] out={out_path}", flush=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": f"generate_ocr_dataset/{__version__} (synthetic OCR dataset preparation)"
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        total_raw = response.headers.get("Content-Length")
        total = int(total_raw) if total_raw and total_raw.isdigit() else 0
        written = 0
        t0 = time.perf_counter()
        last_print = t0
        with tmp_path.open("wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
                now_t = time.perf_counter()
                if now_t - last_print >= 2.0:
                    last_print = now_t
                    speed = written / max(0.001, now_t - t0) / (1024 * 1024)
                    if total:
                        pct = written * 100.0 / total
                        print(f"[download] {written:,}/{total:,} bytes {pct:.1f}% {speed:.2f} MiB/s", flush=True)
                    else:
                        print(f"[download] {written:,} bytes {speed:.2f} MiB/s", flush=True)
    tmp_path.replace(out_path)
    print(f"[download] done {out_path} size={out_path.stat().st_size:,}", flush=True)
    return out_path




def write_run_config(args, families: dict[str, list[str]], output_dir: Path) -> None:
    cfg = vars(args).copy()
    cfg["font_family_count"] = len(families)
    cfg["font_file_count"] = sum(len(v) for v in families.values())
    cfg["created_at"] = now()
    (output_dir / "generator_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def warn_resume_config_drift(args, output_dir: Path) -> None:
    old_path = output_dir / "generator_config.json"
    if not args.resume or not old_path.exists():
        return
    try:
        old = json.loads(old_path.read_text(encoding="utf-8"))
    except Exception:
        print(f"[resume] warning: could not parse previous config at {old_path}", flush=True)
        return
    important = [
        "seed",
        "use_default_sources",
        "source",
        "fonts_dir",
        "extra_font_dir",
        "include_windows_fonts",
        "include_web_fonts",
        "no_dedupe_font_files",
        "no_group_font_families",
        "text_cleanup_profile",
        "strip_arabic_marks",
        "normalize_arabic_yeh_nonfinal",
        "normalize_zwnj",
        "max_lines",
        "val_count",
        "val_ratio",
    ]
    drift = []
    current = vars(args)
    for key in important:
        if old.get(key) != current.get(key):
            drift.append((key, old.get(key), current.get(key)))
    if drift:
        print("[resume] warning: config drift detected. Resume is safest with identical source/font/split settings.", flush=True)
        for key, old_value, new_value in drift[:20]:
            print(f"[resume] drift {key}: old={old_value!r} new={new_value!r}", flush=True)


def configure_stdio() -> None:
    """Force UTF-8 on stdout/stderr.

    Labels, source titles and CLI help text all contain Arabic-script
    characters. On Windows the default console codepage is often cp1252, which
    raises UnicodeEncodeError the moment any of that text reaches the terminal.
    All file I/O in this script already passes an explicit encoding, so this only
    affects console output.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate_ocr_dataset.py",
        description=(
            "Generate synthetic OCR line crops for text-recognition training. "
            "Text is rendered with headless Chrome/Edge using real font files, then "
            "degraded with a configurable scanner/photocopy pipeline. Output is "
            "PaddleOCR-ready: images/, train_list.txt, val_list.txt and metadata.jsonl."
        ),
        epilog=(
            "Text sources: --source PATH:TYPE (repeatable), where TYPE is one of "
            + ", ".join(SOURCE_TYPES)
            + ". Point --source at any file in one of these shapes; no particular "
            "corpus or website is built in. To render an existing label list instead "
            "of raw documents, use --label-source-manifest FILE."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--output-dir", default=None, help="Output directory. Images/manifests/metadata are written here.")
    parser.add_argument("--fonts-dir", default=None, help="Directory containing .ttf/.otf/.woff/.woff2 fonts.")
    parser.add_argument("--extra-font-dir", action="append", default=[], help="Additional read-only font directory. Repeatable.")
    parser.add_argument("--include-windows-fonts", action="store_true", help="Read common Windows fonts from C:/Windows/Fonts or /mnt/c/Windows/Fonts if present.")
    parser.add_argument("--include-web-fonts", action="store_true", help="Also accept .woff/.woff2 web fonts. Default uses desktop .ttf/.otf/.ttc/.otc only.")
    parser.add_argument("--no-dedupe-font-files", action="store_true", help="Do not remove duplicate font files by content hash.")
    parser.add_argument("--no-group-font-families", action="store_true", help="Keep one family per file instead of grouping variants by normalized name.")
    parser.add_argument("--font-include-regex", default=None, help="Only keep font families/paths matching this Python regex. Empty means keep all fonts.")
    parser.add_argument("--font-exclude-regex", default=None, help="Drop font families/paths matching this Python regex after include filtering. Empty means drop none.")
    parser.add_argument("--source", action="append", default=[], help=f"Input file as PATH:TYPE, repeatable. TYPE is one of: {', '.join(SOURCE_TYPES)}")
    parser.add_argument("--use-default-sources", action="store_true", help="Also auto-discover Wikipedia XML dumps inside --default-sources-dir.")
    parser.add_argument("--default-sources-dir", default="./sources", help="Directory scanned for Wikipedia dumps when --use-default-sources is set.")
    parser.add_argument("--convert-latin-kurdish-to-arabic", action="store_true", help="Transliterate Latin-script Kurdish (Kurmanji) source text into Arabic script. Requires the optional asosoft package; ignored without it.")
    parser.add_argument("--max-lines", type=int, default=0, help="Exact maximum generated line images. 0 means run until sources are exhausted.")
    parser.add_argument("--seed", type=int, default=20260629, help="Random seed for reproducible layout/degradation/split choices.")
    parser.add_argument("--target-unit", choices=["line", "phrase48", "short_phrase", "word"], default="line", help="Text chunk type to generate before rendering. line preserves the original broad 6-260 character line generator.")
    parser.add_argument("--label-source-manifest", default=None, help="Optional existing manifest/plain-label file. When set, labels are rendered directly instead of sampling text sources.")
    parser.add_argument("--label-source-repeat", type=int, default=1, help="How many independent visual renders to create for each accepted label from --label-source-manifest.")
    parser.add_argument("--label-source-shuffle", action="store_true", help="Shuffle labels from --label-source-manifest using --seed before rendering.")
    parser.add_argument("--label-source-shuffle-seed", type=int, default=None, help="Optional fixed shuffle seed for --label-source-shuffle. Use this to render the same label order with different render seeds.")
    parser.add_argument("--no-label-source-dedupe", dest="label_source_dedupe", action="store_false", help="Keep duplicate labels from --label-source-manifest. Default dedupes exact normalized labels.")
    parser.set_defaults(label_source_dedupe=True)
    parser.add_argument(
        "--script-filter",
        choices=["any", "arabic_no_latin", "arabic_no_latin_digit", "arabic_letters_marks_space"],
        default="any",
        help="Accept generated text units by script content. arabic_no_latin_digit keeps Arabic-script rows while rejecting Latin letters and digits.",
    )
    parser.add_argument("--line-words-min", type=int, default=1, help="Minimum words per generated line unit when --target-unit line.")
    parser.add_argument("--line-words-max", type=int, default=34, help="Maximum words per generated line unit when --target-unit line.")
    parser.add_argument("--line-words-mode", type=int, default=18, help="Most likely word count for triangular sampling when --target-unit line.")
    parser.add_argument("--line-chars-min", type=int, default=6, help="Minimum characters per generated line unit when --target-unit line.")
    parser.add_argument("--line-chars-max", type=int, default=260, help="Maximum characters per generated line unit when --target-unit line.")
    parser.add_argument("--phrase48-words-min", type=int, default=2, help="Minimum words per generated phrase48 unit.")
    parser.add_argument("--phrase48-words-max", type=int, default=10, help="Maximum words per generated phrase48 unit.")
    parser.add_argument("--phrase48-words-mode", type=int, default=5, help="Most likely word count for triangular sampling when --target-unit phrase48.")
    parser.add_argument("--phrase48-chars-min", type=int, default=16, help="Minimum characters per generated phrase48 unit.")
    parser.add_argument("--phrase48-chars-max", type=int, default=48, help="Maximum characters per generated phrase48 unit.")
    parser.add_argument("--short-phrase-words-min", type=int, default=1, help="Minimum words per generated short_phrase unit.")
    parser.add_argument("--short-phrase-words-max", type=int, default=6, help="Maximum words per generated short_phrase unit.")
    parser.add_argument("--short-phrase-words-mode", type=int, default=3, help="Most likely word count for triangular sampling when --target-unit short_phrase.")
    parser.add_argument("--short-phrase-chars-min", type=int, default=8, help="Minimum characters per generated short_phrase unit.")
    parser.add_argument("--short-phrase-chars-max", type=int, default=32, help="Maximum characters per generated short_phrase unit.")
    parser.add_argument("--word-words-min", type=int, default=1, help="Minimum words per generated word unit.")
    parser.add_argument("--word-words-max", type=int, default=1, help="Maximum words per generated word unit.")
    parser.add_argument("--word-words-mode", type=int, default=1, help="Most likely word count for triangular sampling when --target-unit word.")
    parser.add_argument("--word-chars-min", type=int, default=2, help="Minimum characters per generated word unit.")
    parser.add_argument("--word-chars-max", type=int, default=24, help="Maximum characters per generated word unit.")
    parser.add_argument("--pseudo-kurdish", action="store_true", help="Shuffle words inside paragraphs. Default is off because it weakens realistic line distribution.")
    parser.add_argument("--max-seen-docs", type=int, default=200000, help="Dedup memory for source document IDs.")
    parser.add_argument("--resume", action="store_true", help="Force append to an existing output directory. Existing rows are auto-resumed even without this flag unless --overwrite is used.")
    parser.add_argument("--overwrite", action="store_true", help="Start fresh by deleting this generator's managed files in output-dir: images, manifests, metadata, config, and temp HTML.")
    parser.add_argument("--download-wiki-dump", action="append", default=[], metavar="CODE", help=f"Download a Wikipedia dump into --download-dir, then exit unless generation options are also supplied. Known codes: {', '.join(WIKI_DUMP_URLS)}. Repeatable.")
    parser.add_argument("--download-dir", default="./downloads", help="Destination directory for --download-wiki-dump.")
    parser.add_argument("--wiki-dump-url", default=None, help="Fetch this dump URL instead of a known code. The file name is taken from the URL path.")
    parser.add_argument("--text-cleanup-profile", choices=["none", "conservative_sorani"], default="conservative_sorani", help="Unicode cleanup profile applied to generated labels.")
    parser.add_argument("--strip-arabic-marks", action="store_true", help="Delete Arabic combining marks/diacritics from generated labels.")
    parser.add_argument("--normalize-arabic-yeh-nonfinal", action="store_true", help="Map ي to ی only when followed by another Arabic letter.")
    parser.add_argument("--normalize-zwnj", action="store_true", help="Delete ZERO WIDTH NON-JOINER. Default keeps it, which is safer for Persian.")
    parser.add_argument("--mode", choices=["sheet"], default="sheet", help="Compatibility option. Line crops are always rendered as batched browser sheets.")
    parser.add_argument("--renderer", choices=["chrome"], default="chrome", help="Text rendering engine. chrome uses Chrome/Edge sheet screenshots.")
    parser.add_argument("--fallback-renderer", choices=["none", "pil"], default="none", help="What to do when a single-line browser capture still fails: none re-raises, pil renders that line with Pillow instead of losing it.")
    parser.add_argument("--browser-exe", default=None, help="Chrome/Edge executable. Default searches common Chrome/Edge paths.")
    parser.add_argument("--browser-timeout", type=int, default=60, help="Seconds to wait for each browser screenshot.")
    parser.add_argument("--browser-retries", type=int, default=0, help="Retry count after a failed Chrome/Edge screenshot. Total attempts are 1 + this value. Default 0 prevents repeated Chrome relaunches during unattended runs.")
    parser.add_argument("--browser-retry-delay", type=float, default=1.0, help="Seconds to wait between failed Chrome/Edge screenshot attempts.")
    parser.add_argument("--browser-screenshot-settle-seconds", type=float, default=10.0, help="After Chrome exits, wait up to this many seconds for the screenshot file to appear and become non-empty.")
    parser.add_argument("--browser-screenshot-settle-poll", type=float, default=0.10, help="Seconds between screenshot-file existence checks during --browser-screenshot-settle-seconds.")
    parser.add_argument("--browser-verbose-log", action="store_true", help="Add Chrome --v=1 verbose logging. Off by default because it produces large histogram spam.")
    parser.add_argument("--browser-max-screenshot-height", type=int, default=16000, help="Fail early if a sheet screenshot would exceed this pixel height. 0 disables this guard.")
    parser.add_argument("--sheet-lines", type=int, default=64, help="Number of independent text lines rendered per browser screenshot.")
    parser.add_argument("--sheet-workers", type=int, default=2, help="Parallel Chrome/Edge screenshot workers.")
    parser.add_argument("--crop-pad-x", type=int, default=None, help="Compatibility override: fixed horizontal crop padding. Prefer crop-pad-x-min/max.")
    parser.add_argument("--crop-pad-y", type=int, default=None, help="Compatibility override: fixed vertical crop padding. Prefer crop-pad-y-min/max.")
    parser.add_argument("--crop-pad-x-min", type=int, default=4, help="Minimum random horizontal pixels kept around detected text ink.")
    parser.add_argument("--crop-pad-x-max", type=int, default=28, help="Maximum random horizontal pixels kept around detected text ink.")
    parser.add_argument("--crop-pad-y-min", type=int, default=2, help="Minimum random vertical pixels kept around detected text ink.")
    parser.add_argument("--crop-pad-y-max", type=int, default=12, help="Maximum random vertical pixels kept around detected text ink.")
    parser.add_argument("--reject-crop-width-over", type=int, default=0, help="Reject rendered crops wider than this many pixels after trimming/degradation. 0 disables this fail-closed filter.")
    parser.add_argument("--writer-workers", type=int, default=4, help="Parallel image file writer workers.")
    parser.add_argument("--tmp-dir", default=None, help="HTML temp directory. Default: OUTPUT_DIR/_tmp_html.")
    parser.add_argument("--bench-every", type=int, default=100, help="Print benchmark summary every N completed pages.")
    parser.add_argument("--val-ratio", type=float, default=0.002, help="Validation split probability when --val-count is not set.")
    parser.add_argument("--val-count", type=int, default=None, help="Exact validation row count, randomly distributed by line ID. Requires --max-lines.")
    parser.add_argument(
        "--degradation-profile",
        default="mixed",
        choices=["mixed", "browser_clean", "digital_clean", "raster_pdf", "book_scan", "thick_scan", "thin_scan", "overcooked_scan", "photocopy_bleed", "binary_lowres_bloat", "faded_scan", "ugly_scan"],
        help="Line-image degradation profile. mixed samples several realistic profiles.",
    )
    parser.add_argument("--jpeg-quality-min", type=int, default=62, help="Lowest JPEG quality used inside degradation pass.")
    parser.add_argument("--jpeg-quality-max", type=int, default=92, help="Highest JPEG quality used inside degradation pass.")
    parser.add_argument("--perspective-warp-prob", type=float, default=0.06, help="Probability that a line crop gets a mild corner warp.")
    parser.add_argument("--max-perspective-shift-frac", type=float, default=0.025, help="Maximum corner movement as a fraction of the smaller crop side.")
    parser.add_argument("--geometry-pad-px", type=int, default=32, help="White padding added before warp/rotation so text is not clipped.")
    parser.add_argument("--max-rotation-degrees", type=float, default=0.8, help="Maximum absolute mild rotation angle for selected non-clean crops.")
    parser.add_argument("--scanner-band-prob", type=float, default=0.07, help="Probability of adding faint wave/banding brightness artifacts.")
    parser.add_argument("--edge-shadow-prob", type=float, default=0.07, help="Probability of adding a mild dark edge shadow.")
    parser.add_argument("--stain-prob", type=float, default=0.035, help="Probability of adding faint brown paper stains.")
    parser.add_argument("--extra-noise-prob", type=float, default=0.16, help="Probability of adding mild full-image sensor/scanner noise.")
    parser.add_argument("--nearby-rule-prob", type=float, default=0.06, help="Probability of adding a nearby horizontal/vertical rule fragment.")
    parser.add_argument("--text-opacity-prob", type=float, default=0.08, help="Probability that browser-rendered text is slightly faded before screenshot.")
    parser.add_argument("--word-spacing-prob", type=float, default=0.22, help="Probability that a line gets non-default spacing between words.")
    parser.add_argument("--word-spacing-min-px", type=float, default=-0.8, help="Minimum word spacing in pixels when word spacing is applied.")
    parser.add_argument("--word-spacing-max-px", type=float, default=4.0, help="Maximum word spacing in pixels when word spacing is applied.")
    parser.add_argument("--letter-spacing-prob", type=float, default=0.035, help="Probability that a line gets tiny spacing between glyphs; kept rare to avoid breaking Arabic-script joining.")
    parser.add_argument("--letter-spacing-min-px", type=float, default=0.05, help="Minimum letter spacing in pixels when letter spacing is applied.")
    parser.add_argument("--letter-spacing-max-px", type=float, default=0.35, help="Maximum letter spacing in pixels when letter spacing is applied.")
    parser.add_argument("--heavy-blur-prob", type=float, default=0.05, help="Probability of adding stronger blur than the normal scan profile.")
    parser.add_argument("--low-quality-jpeg-prob", type=float, default=0.08, help="Probability of using lower JPEG quality than the normal range.")
    parser.add_argument("--low-quality-jpeg-min", type=int, default=42, help="Lowest JPEG quality for low-quality JPEG cases.")
    parser.add_argument("--low-quality-jpeg-max", type=int, default=68, help="Highest JPEG quality for low-quality JPEG cases.")
    parser.add_argument("--photocopy-bleed-intensity-min", type=float, default=0.20, help="Minimum intensity for photocopy_bleed stroke expansion/pixelation. 0.0 is mild, 1.0 is severe.")
    parser.add_argument("--photocopy-bleed-intensity-max", type=float, default=0.82, help="Maximum intensity for photocopy_bleed stroke expansion/pixelation. 0.0 is mild, 1.0 is severe.")
    parser.add_argument("--binary-bloat-intensity-min", type=float, default=0.10, help="Minimum intensity for binary_lowres_bloat low-DPI thresholded glyph expansion. 0.0 is mild, 1.0 is severe.")
    parser.add_argument("--binary-bloat-intensity-max", type=float, default=0.42, help="Maximum intensity for binary_lowres_bloat low-DPI thresholded glyph expansion. 0.0 is mild, 1.0 is severe.")
    parser.add_argument("--dry-run", action="store_true", help="Validate sources/fonts/config and exit without rendering pages or lines.")
    return parser.parse_args()


def main() -> None:
    configure_stdio()
    args = parse_args()

    download_only = bool(args.download_wiki_dump) or bool(args.wiki_dump_url)
    if download_only:
        urls: list[str] = []
        for code in args.download_wiki_dump:
            url = WIKI_DUMP_URLS.get(code.strip().lower())
            if url is None:
                raise ValueError(
                    f"unknown dump code {code!r}. Known codes: {', '.join(WIKI_DUMP_URLS)}. "
                    "For any other wiki, pass --wiki-dump-url instead."
                )
            urls.append(url)
        if args.wiki_dump_url:
            urls.append(args.wiki_dump_url)
        for url in urls:
            name = Path(urllib.parse.urlparse(url).path).name
            if not name:
                raise ValueError(f"cannot derive a file name from dump URL: {url}")
            download_with_progress(url, Path(args.download_dir) / name)
        if args.dry_run or not args.output_dir or not args.fonts_dir:
            print("[done] download finished.", flush=True)
            return

    if not args.output_dir:
        raise ValueError("--output-dir is required unless only downloading a wiki dump")
    if not args.fonts_dir:
        raise ValueError("--fonts-dir is required unless only downloading a wiki dump")
    if args.convert_latin_kurdish_to_arabic and asosoft is None:
        print(
            "[warn] --convert-latin-kurdish-to-arabic needs the optional 'asosoft' package, "
            "which is not installed. Source text is used unchanged.",
            flush=True,
        )
    if args.label_source_manifest:
        # Labels come from an existing manifest, so raw documents are not needed.
        pass
    elif not (args.source or args.use_default_sources):
        raise ValueError(
            "No text source given. Pass --source PATH:TYPE (repeatable), "
            "--use-default-sources to scan --default-sources-dir, "
            "or --label-source-manifest FILE to render an existing label list."
        )
    if args.crop_pad_x is not None:
        args.crop_pad_x_min = args.crop_pad_x
        args.crop_pad_x_max = args.crop_pad_x
    if args.crop_pad_y is not None:
        args.crop_pad_y_min = args.crop_pad_y
        args.crop_pad_y_max = args.crop_pad_y
    for prefix in ("line", "phrase48", "short_phrase", "word"):
        min_words = getattr(args, f"{prefix}_words_min")
        max_words = getattr(args, f"{prefix}_words_max")
        mode_words = getattr(args, f"{prefix}_words_mode")
        min_chars = getattr(args, f"{prefix}_chars_min")
        max_chars = getattr(args, f"{prefix}_chars_max")
        if min_words < 1 or max_words < 1:
            raise ValueError(f"--{prefix.replace('_', '-')}-words-min/max must be at least 1.")
        if min_words > max_words:
            raise ValueError(f"--{prefix.replace('_', '-')}-words-min cannot be larger than max.")
        if not (min_words <= mode_words <= max_words):
            raise ValueError(f"--{prefix.replace('_', '-')}-words-mode must be between min and max.")
        if min_chars < 1 or max_chars < 1:
            raise ValueError(f"--{prefix.replace('_', '-')}-chars-min/max must be at least 1.")
        if min_chars > max_chars:
            raise ValueError(f"--{prefix.replace('_', '-')}-chars-min cannot be larger than max.")
    if args.crop_pad_x_min < 0 or args.crop_pad_y_min < 0:
        raise ValueError("Crop padding ranges must be non-negative.")
    if args.crop_pad_x_min > args.crop_pad_x_max:
        raise ValueError("--crop-pad-x-min cannot be larger than --crop-pad-x-max.")
    if args.crop_pad_y_min > args.crop_pad_y_max:
        raise ValueError("--crop-pad-y-min cannot be larger than --crop-pad-y-max.")
    if args.reject_crop_width_over < 0:
        raise ValueError("--reject-crop-width-over must be 0 or a positive pixel width.")
    if args.label_source_repeat < 1:
        raise ValueError("--label-source-repeat must be at least 1.")
    if args.photocopy_bleed_intensity_min < 0 or args.photocopy_bleed_intensity_max > 1:
        raise ValueError("Photocopy bleed intensity bounds must be between 0.0 and 1.0.")
    if args.photocopy_bleed_intensity_min > args.photocopy_bleed_intensity_max:
        raise ValueError("--photocopy-bleed-intensity-min cannot be larger than --photocopy-bleed-intensity-max.")
    if args.binary_bloat_intensity_min < 0 or args.binary_bloat_intensity_max > 1:
        raise ValueError("Binary bloat intensity bounds must be between 0.0 and 1.0.")
    if args.binary_bloat_intensity_min > args.binary_bloat_intensity_max:
        raise ValueError("--binary-bloat-intensity-min cannot be larger than --binary-bloat-intensity-max.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.tmp_dir is None:
        args.tmp_dir = str(output_dir / "_tmp_html")
    metadata_rows_existing, manifest_rows_existing, max_line_id = managed_output_row_state(output_dir)
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume conflict. Use --overwrite to start fresh or omit it to auto-resume.")
    if args.overwrite:
        clear_managed_output(output_dir)
        metadata_rows_existing, manifest_rows_existing, max_line_id = 0, 0, -1
        print(f"[overwrite] cleared managed generator output in {output_dir}", flush=True)
    elif not args.resume and (metadata_rows_existing > 0 or manifest_rows_existing > 0):
        args.resume = True
        print(
            f"[resume] auto-enabled because existing rows were found in output-dir. "
            f"metadata_rows={metadata_rows_existing} manifest_rows={manifest_rows_existing} max_line_id={max_line_id}. "
            f"Use --overwrite to start fresh.",
            flush=True,
        )
    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = args.tmp_dir

    random.seed(args.seed)
    extra_font_dirs = [Path(p) for p in args.extra_font_dir]
    if args.include_windows_fonts:
        extra_font_dirs.extend(windows_font_dirs())
    font_extensions = {".ttf", ".otf", ".ttc", ".otc"}
    if args.include_web_fonts:
        font_extensions.update({".woff", ".woff2"})
    families = load_fonts(
        Path(args.fonts_dir),
        extra_font_dirs,
        font_extensions,
        dedupe_font_files=not args.no_dedupe_font_files,
        group_font_families=not args.no_group_font_families,
        font_include_regex=args.font_include_regex,
        font_exclude_regex=args.font_exclude_regex,
    )
    sources = source_specs(args)
    print_source_availability(sources)

    print("[config] effective settings:", flush=True)
    for key in sorted(vars(args)):
        print(f"[config] {key}={getattr(args, key)}", flush=True)
    print(f"[config] source_count={len(sources)} font_families={len(families)} font_files={sum(len(v) for v in families.values())}", flush=True)
    if args.max_lines:
        print(f"[plan] image_count={args.max_lines} because --max-lines is set; each image is one OCR line crop.", flush=True)
    else:
        print("[plan] image_count=unbounded because --max-lines=0; generation runs until text sources are exhausted.", flush=True)
    if args.val_count is not None:
        print(f"[split] mode=random_exact_by_line_id val_rows={args.val_count} train_rows={args.max_lines - args.val_count}", flush=True)
    else:
        print(f"[split] mode=random_ratio_per_row val_ratio={args.val_ratio}", flush=True)
    print("[status] train_list means image-relative-path<TAB>label for PaddleOCR training.", flush=True)
    print("[status] val_list means the same format for validation rows.", flush=True)
    print("[status] metadata.jsonl stores one JSON object per generated line with font/source/degradation/bbox evidence.", flush=True)

    if args.resume and manifest_rows_existing > 0 and metadata_rows_existing == 0:
        raise RuntimeError(
            "Cannot safely resume: train/val manifests exist but metadata.jsonl has no rows. "
            "Use --overwrite to start fresh, or restore metadata.jsonl before resuming."
        )
    if args.resume and metadata_rows_existing != manifest_rows_existing:
        print(
            f"[resume] warning: metadata_rows={metadata_rows_existing} manifest_rows={manifest_rows_existing}. "
            "Resume will skip using metadata.jsonl line_id evidence.",
            flush=True,
        )
    args.resume_start_line_id = max_line_id + 1 if args.resume else 0
    if args.resume:
        print(
            f"[resume] enabled existing_metadata_rows={metadata_rows_existing} existing_manifest_rows={manifest_rows_existing} max_line_id={max_line_id} "
            f"resume_start_line_id={args.resume_start_line_id}",
            flush=True,
        )
    warn_resume_config_drift(args, output_dir)
    write_run_config(args, families, output_dir)
    if args.dry_run:
        print("[done] dry_run=true; no pages rendered and no dataset rows written.", flush=True)
        return

    writer = ManifestWriter(output_dir, args.val_ratio, args.val_count, args.writer_workers, args.seed, args.max_lines, args.resume)
    bench = Bench(args.bench_every)
    completed_ok = False
    try:
        run_browser_sheet(args, families, writer, bench)
        completed_ok = True
    finally:
        writer.close()
        if bench.pages:
            bench.print_and_reset()
        status = "done" if completed_ok else "stopped"
        print(
            f"[{status}] rows_total={writer.total_rows} train_rows={writer.train_rows} val_rows={writer.val_rows} "
            f"images_dir={writer.images_dir} train_list={writer.manifest_train} val_list={writer.manifest_val}",
            flush=True,
        )


if __name__ == "__main__":
    configure_stdio()
    try:
        main()
    except KeyboardInterrupt:
        print("\n[abort] interrupted by user. Generated rows are already on disk; re-run to resume.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        # Bad configuration and environment problems are the user's to fix, not
        # bugs. Report them as a single readable line instead of a traceback.
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        print("[error] re-run with --help to check the available options.", file=sys.stderr)
        sys.exit(1)
