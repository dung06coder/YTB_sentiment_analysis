"""
preprocessing.py -- Text cleaning for Vietnamese YouTube comments
===================================================================

Main entry points
-----------------
    clean_text(text, segment=False)          -> str    (one comment)
    preprocess_dataframe(df, ...)            -> DataFrame (whole CSV)

Command line
------------
    python preprocessing.py                       # run the demo
    python preprocessing.py --input raw.csv --output clean.csv --text-col text
    python preprocessing.py --input raw.csv --output clean.csv --segment

Pipeline order (the order matters!)
-----------------------------------
 1. Unicode NFC normalisation, HTML unescape, strip zero-width characters
 2. Remove HTML tags, URLs, e-mails, @mentions, video timestamps
 3. Lowercase
 4. ASCII emoticons (":))", "<3", "-_-") -> Vietnamese words
 5. Emoji (😂, 😡, ...) -> Vietnamese words
 6. Remove remaining punctuation / special characters
 7. Per-token: normalise laughter, squash stretched letters, expand teencode
 8. (optional) Vietnamese word segmentation for PhoBERT / word-level TF-IDF
 9. Collapse whitespace
"""

from __future__ import annotations

import argparse
import html
import re
import unicodedata
from typing import Optional

__all__ = ["clean_text", "preprocess_dataframe", "EMOJI_MAP", "TEENCODE"]


# =====================================================================
# 1. DICTIONARIES  (edit freely -- add anything you see in real comments)
# =====================================================================

# Written as {Vietnamese word: "all emoji that mean it"} to keep it compact.
_EMOJI_GROUPS = {
    # ---- positive ----
    "cười": "😂🤣😆😹",
    "vui": "😀😃😄😁😊🙂☺🙃😺😸🤗",
    "yêu": "😍🥰😘😗😙😚😻😽❤💕💖💗💓💞💘💝💙💚💛🧡💜🖤🤍🤎♥🫶",
    "tuyệt vời": "🤩👏🙌💯🔥⭐🌟🏆",
    "tốt": "👍👌🤝✌💪🆗✅",
    "ngầu": "😎",
    "chúc mừng": "🎉🥳🎊🎂",
    "cảm ơn": "🙏",
    "ngon": "😋",
    "đùa": "😉😜😝😛🤪",
    "xúc động": "🥹",
    # ---- negative ----
    "tức giận": "😡😠🤬👿😤💢",
    "khóc": "😭😢😿",
    "buồn": "😞😔☹🙁😟😕😩😫😣😖😪",
    "tội nghiệp": "🥺",
    "đau lòng": "💔",
    "tệ": "👎",
    "chửi": "🖕",
    "ghê tởm": "🤮🤢",
    "rác": "💩",
    "chán": "😒🙄😑😴🥱🤦",
    "sợ": "😱😨😰😧😦",
    # ---- neutral / ambiguous ----
    "ngạc nhiên": "😮😲😯🤯😳",
    "suy nghĩ": "🤔🧐",
    "hề": "🤡",
    "mỉa mai": "😏",
    "bình thường": "😐😶",
    "không biết": "🤷",
    "ngượng": "😅😬🙈",
    "lo lắng": "😥😓",
    "chết cười": "💀☠",  # Gen-Z usage: "dying of laughter"
}

# Emoji not listed above are simply removed (replaced by a space).
EMOJI_MAP = {
    ch: word
    for word, chars in _EMOJI_GROUPS.items()
    for ch in chars
    if ch not in "\ufe0f\ufe0e\u200d"
}

# {expanded form: "space separated list of abbreviations"}
# Only WHOLE tokens are replaced, so "5k" or "kho" are never touched.
_TEENCODE_GROUPS = {
    "không": "ko k kh khong khg hok hong hông hem k0 kô",
    "không phải": "kp kphai kphải",
    "không biết": "kbiet kbiết",
    "không có": "kco",
    "được": "dc đc dk đk dzc duoc",
    "với": "vs",
    "gì": "j",
    "vậy": "z zậy zay",
    "vui": "zui",
    "vô": "zô",
    "rồi": "r rùi ròi roài",
    "quá": "wa wá qá",
    "cũng": "cx cg cug cũg",
    "nhưng": "nhg nhưg",
    "nhưng mà": "nma nhma",
    "mình": "mk mik mjnh mih",
    "mọi người": "mn mng",
    "người": "ng ngừoi",
    "người ta": "ngta",
    "người yêu": "ny ngiu",
    "bình thường": "bt bthg bth",
    "biết": "bik bít bit",
    "thích": "thik thix thick",
    "yêu": "iu",
    "cảm ơn": "tks thks thanks thank thankiu camon",
    "xin lỗi": "sr sry sorry",
    "bình luận": "cmt cmts comment",
    "video": "vid vdeo vdo",
    "anh em": "ae",
    "bao nhiêu": "bn bnh",
    "trả lời": "tl trl",
    "điện thoại": "đt dt dthoai",
    "hôm nay": "hnay hnai",
    "hôm qua": "hqua",
    "trước": "trc",
    "bây giờ": "bh bgio",
    "nữa": "nx",
    "mới": "ms",
    "làm": "lm",
    "luôn": "lun",
    "liên quan": "lq",
    "ví dụ": "vd",
    "như thế nào": "ntn",
    "nói chuyện": "nch",
    "nhắn tin": "nt",
    "sao": "s",
    "tại sao": "tsao",
    "đang": "dg đg đag",
    "ok": "oke okê okie okay okey ôk ôkê",
    "làm ơn": "pls plz",
    "kết bạn": "kb",
    "dễ thương": "dthw dthuong",
    "xinh": "xink",
    "nhiều": "nhìu nhiu",
    "bé": "pé",
    "bạn": "pạn",
    "trai": "zai",
    # Profanity is kept (it carries sentiment, e.g. "hay vãi" / "dở đm"),
    # but spelling variants are merged into one token.
    "đm": "dm dmm đmm đcm dcm",
    "vãi": "vl vcl vkl",
}
TEENCODE = {
    abbr: full
    for full, abbrs in _TEENCODE_GROUPS.items()
    for abbr in abbrs.split()
}


# =====================================================================
# 2. REGEXES (compiled once)
# =====================================================================

_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\ufe0e\ufe0f]")
# Real tags start with a letter or "/", so "<3" (a heart) is NOT treated as a tag.
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|vn|be|ly|tv|io|me|info)\b(?:/\S*)?",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_MENTION_RE = re.compile(r"@\S+")
_TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")  # "3:45", "1:02:30"
_PUNCT_RE = re.compile(r"[^\w\s]|_")  # anything that is not letter/digit/space
_SPACES_RE = re.compile(r"\s+")

# Order matters: longer / more specific patterns first. Text is already lowercase.
_B = r"(?<!\w)"  # "not preceded by a word character"
_E = r"(?!\w)"   # "not followed by a word character"
_EMOTICONS = [
    (r"</3", "đau lòng"),
    (r"<3+", "yêu"),
    (r":'\(+", "khóc"),
    (r">:\(+|:@", "tức giận"),
    (r"-_-|:\||:-\|", "chán"),
    (rf"{_B}t_t{_E}|{_B}t\.t{_E}|;_;|:<", "buồn"),
    (r":-?\(+|=\(+", "buồn"),
    (r":-?\)+|=\)+|\(:|\^_*\^", "vui"),
    (rf":-?d+{_E}|=d+{_E}|{_B}xd+{_E}", "cười"),
    (rf":v+{_E}|:3+{_E}|:-?p+{_E}|;-?\)+", "đùa"),
    (rf":-?o{_E}|{_B}o_o{_E}", "ngạc nhiên"),
]
_EMOTICON_RES = [(re.compile(p), rf" {w} ") for p, w in _EMOTICONS]

# Laughter ("haha", "hihi", "kkkk", "hjhj") -> one token. "huhu" means crying.
_LAUGH_RE = re.compile(r"k{2,}|a?(?:ha|he|hi|hj){2,}h?|h{3,}")
_CRY_RE = re.compile(r"a?(?:hu){2,}h?")
# 3+ repeated LETTERS ("hayyyy", "quááááá") -> one letter. Digits are untouched.
_REPEAT_RE = re.compile(r"([^\W\d_])\1{2,}")

_SYMBOL_CATEGORIES = {"So", "Sk"}  # emoji, pictographs, modifier symbols

_word_tokenize = None  # lazy-loaded underthesea function


# =====================================================================
# 3. INDIVIDUAL STEPS
# =====================================================================

def convert_emoticons(text: str) -> str:
    """':))' -> ' vui ', '<3' -> ' yêu ', '-_-' -> ' chán '."""
    for pattern, repl in _EMOTICON_RES:
        text = pattern.sub(repl, text)
    return text


def convert_emoji(text: str) -> str:
    """Known emoji -> Vietnamese word; unknown emoji/symbols -> a space."""
    out = []
    for ch in text:
        word = EMOJI_MAP.get(ch)
        if word is not None:
            out.append(f" {word} ")
        elif unicodedata.category(ch) in _SYMBOL_CATEGORIES:
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def normalize_tokens(text: str) -> str:
    """Laughter, stretched letters and teencode, token by token."""
    out = []
    for tok in text.split():
        if _LAUGH_RE.fullmatch(tok):
            out.append("haha")
            continue
        if _CRY_RE.fullmatch(tok):
            out.append("khóc")
            continue
        tok = _REPEAT_RE.sub(r"\1", tok)  # "koooo" -> "ko"
        out.append(TEENCODE.get(tok, tok))  # "ko"    -> "không"
    return " ".join(out)


def segment_vietnamese(text: str) -> str:
    """
    Word segmentation: "học máy rất hay" -> "học_máy rất hay".
    Needs `pip install underthesea`. Compound words are joined with "_".
    """
    global _word_tokenize
    if _word_tokenize is None:
        try:
            from underthesea import word_tokenize
        except ImportError as exc:
            raise ImportError(
                "segment=True needs underthesea: pip install underthesea"
            ) from exc
        _word_tokenize = word_tokenize
    return _word_tokenize(text, format="text")


# =====================================================================
# 4. MAIN API
# =====================================================================

def clean_text(text, segment: bool = False) -> str:
    """
    Clean ONE raw comment and return normalised lowercase text.

    Parameters
    ----------
    text : str | None | float
        Raw comment. None / NaN / non-strings are handled (returns "" for empty).
    segment : bool
        If True, also run Vietnamese word segmentation (adds "_" inside
        compound words). Use the SAME value in training and in the backend.
    """
    # --- 0. safe input ---------------------------------------------------
    if text is None or (isinstance(text, float) and text != text):  # NaN
        return ""
    if not isinstance(text, str):
        text = str(text)

    # --- 1. unicode / html ----------------------------------------------
    text = unicodedata.normalize("NFC", text)  # composed Vietnamese diacritics
    text = html.unescape(text)                 # "&amp;" -> "&", "&#39;" -> "'"
    text = _ZERO_WIDTH_RE.sub("", text)

    # --- 2. noise removal -------------------------------------------------
    text = _HTML_TAG_RE.sub(" ", text)         # <br>, <a href=...>
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _TIMESTAMP_RE.sub(" ", text)

    # --- 3. lowercase -----------------------------------------------------
    text = text.lower()

    # --- 4/5. emoticons and emoji -> words --------------------------------
    text = convert_emoticons(text)
    text = convert_emoji(text)

    # --- 6. special characters -------------------------------------------
    text = _PUNCT_RE.sub(" ", text)

    # --- 7. token-level normalisation -------------------------------------
    text = normalize_tokens(text)

    # --- 8. optional word segmentation ------------------------------------
    if segment and text:
        text = segment_vietnamese(text)

    # --- 9. whitespace ----------------------------------------------------
    return _SPACES_RE.sub(" ", text).strip()


def preprocess_dataframe(
    df,
    text_col: str = "text",
    out_col: str = "clean_text",
    segment: bool = False,
    drop_empty: bool = True,
    drop_duplicates: bool = True,
):
    """
    Add a cleaned column to a pandas DataFrame.

    The raw `text_col` is kept untouched so you can re-run with a better
    preprocessing later. Rows that are empty after cleaning are dropped, and
    (by default) so are duplicate cleaned comments -- this prevents the same
    comment landing in both train and test (data leakage).
    """
    df = df.copy()
    df[out_col] = df[text_col].map(lambda t: clean_text(t, segment=segment))
    if drop_empty:
        df = df[df[out_col] != ""]
    if drop_duplicates:
        df = df.drop_duplicates(subset=out_col)
    return df.reset_index(drop=True)


# =====================================================================
# 5. DEMO / CLI
# =====================================================================

_DEMO_COMMENTS = [
    "Video hay quá!!! Cảm ơn bạn nhiều 😍😍 https://youtu.be/abc123",
    "ko hiểu sao nhiều ng thích :))",
    'Đoạn <a href="https://youtube.com/watch?v=x&t=45">0:45</a> hài vcl 🤣🤣<br>xem 10 lần vẫn cười',
    "@Minh Tuấn đồng ý vs bạn, nhạc dở tệ 😡😡😡",
    "Phimmmmm chánnnn quááááá -_-",
    "Cảm ơn ad, video rất hữu ích &amp; dễ hiểu ❤️",
    "Hnay mới xem, dc phết kkkkk",
    "Làm ơn đừng làm vậy nữa huhuhu 😭 <3",
    "Giá 5k thôi à? lúc 3:45 mình xem rồi",
    "",
    None,
]


def _demo() -> None:
    print("RAW  ->  CLEAN\n" + "-" * 60)
    for raw in _DEMO_COMMENTS:
        print(f"{raw!r}\n  -> {clean_text(raw)!r}\n")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Clean a CSV of comments.")
    parser.add_argument("--input", help="input CSV path (UTF-8)")
    parser.add_argument("--output", help="output CSV path")
    parser.add_argument("--text-col", default="text", help="raw text column")
    parser.add_argument("--out-col", default="clean_text", help="new column name")
    parser.add_argument("--segment", action="store_true",
                        help="Vietnamese word segmentation (needs underthesea)")
    args = parser.parse_args()

    if not args.input:
        _demo()
        return

    import pandas as pd  # imported here so the module works without pandas

    df = pd.read_csv(args.input, encoding="utf-8")
    before = len(df)
    df = preprocess_dataframe(
        df, text_col=args.text_col, out_col=args.out_col, segment=args.segment
    )
    out_path = args.output or args.input.replace(".csv", "_clean.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Done: {before} rows in, {len(df)} rows out -> {out_path}")


if __name__ == "__main__":
    _main()