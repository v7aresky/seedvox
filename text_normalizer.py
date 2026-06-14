"""
Text normalization for TTS — based on Tacotron2/keithito cleaners.

Handles:
- Number-to-words expansion (123 → "one hundred twenty three")
- Currency ($3.50 → "three dollars fifty cents")
- Ordinals (1st → "first", 2nd → "second")
- Common abbreviations (Mr. → Mister, etc. → et cetera)
- Time expansion (3:15 → "three fifteen")
- Whitespace normalization
- Unicode cleanup

Usage:
    from text_normalizer import normalize_text
    clean = normalize_text("Mr. Smith paid $3.50 for 2 items on Jan. 1st, 2024.")
    # → "mister smith paid three dollars fifty cents for two items on january first, twenty twenty four."
"""

import re
import unicodedata

# ==============================================================================
# NUMBER TO WORDS
# ==============================================================================

_ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
         'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
         'seventeen', 'eighteen', 'nineteen']
_tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']

def _int_to_words(n):
    """Convert integer to English words."""
    if n < 0:
        return 'minus ' + _int_to_words(-n)
    if n == 0:
        return 'zero'
    
    parts = []
    if n >= 1_000_000_000:
        billions = n // 1_000_000_000
        parts.append(_int_to_words(billions) + ' billion')
        n %= 1_000_000_000
    if n >= 1_000_000:
        millions = n // 1_000_000
        parts.append(_int_to_words(millions) + ' million')
        n %= 1_000_000
    if n >= 1000:
        thousands = n // 1000
        parts.append(_int_to_words(thousands) + ' thousand')
        n %= 1000
    if n >= 100:
        hundreds = n // 100
        parts.append(_ones[hundreds] + ' hundred')
        n %= 100
    if n >= 20:
        parts.append(_tens[n // 10])
        if n % 10:
            parts.append(_ones[n % 10])
    elif n > 0:
        parts.append(_ones[n])
    
    return ' '.join(parts)


def _number_to_words(match):
    """Regex replacement function for numbers."""
    text = match.group(0)
    
    # Strip commas from numbers (e.g., 14,941 → 14941)
    text = text.replace(',', '')
    
    # Handle decimals (e.g., 3.14)
    if '.' in text:
        parts = text.split('.')
        integer_part = _int_to_words(int(parts[0]))
        # Read decimal digits individually
        decimal_part = ' '.join(_ones[int(d)] if d != '0' else 'zero' for d in parts[1])
        return f"{integer_part} point {decimal_part}"
    
    # Handle commas in numbers (e.g., 1,000,000)
    text_clean = text.replace(',', '')
    try:
        return _int_to_words(int(text_clean))
    except ValueError:
        return text


# ==============================================================================
# CURRENCY
# ==============================================================================

def _expand_currency(text):
    """Expand currency expressions."""
    # $X.XX
    def _dollar_match(m):
        dollars = int(m.group(1))
        cents = int(m.group(2)) if m.group(2) else 0
        parts = []
        if dollars:
            parts.append(_int_to_words(dollars) + (' dollar' if dollars == 1 else ' dollars'))
        if cents:
            parts.append(_int_to_words(cents) + (' cent' if cents == 1 else ' cents'))
        return ' '.join(parts) if parts else 'zero dollars'
    
    text = re.sub(r'\$(\d+)\.(\d{2})', _dollar_match, text)
    text = re.sub(r'\$(\d+)', lambda m: _int_to_words(int(m.group(1))) + (' dollar' if m.group(1) == '1' else ' dollars'), text)
    
    # £X
    text = re.sub(r'£(\d+)', lambda m: _int_to_words(int(m.group(1))) + (' pound' if m.group(1) == '1' else ' pounds'), text)
    
    # €X
    text = re.sub(r'€(\d+)', lambda m: _int_to_words(int(m.group(1))) + (' euro' if m.group(1) == '1' else ' euros'), text)
    
    return text


# ==============================================================================
# ORDINALS
# ==============================================================================

_ordinal_map = {
    '1st': 'first', '2nd': 'second', '3rd': 'third', '4th': 'fourth',
    '5th': 'fifth', '6th': 'sixth', '7th': 'seventh', '8th': 'eighth',
    '9th': 'ninth', '10th': 'tenth', '11th': 'eleventh', '12th': 'twelfth',
    '13th': 'thirteenth', '14th': 'fourteenth', '15th': 'fifteenth',
    '16th': 'sixteenth', '17th': 'seventeenth', '18th': 'eighteenth',
    '19th': 'nineteenth', '20th': 'twentieth', '21st': 'twenty first',
    '22nd': 'twenty second', '23rd': 'twenty third', '24th': 'twenty fourth',
    '25th': 'twenty fifth', '26th': 'twenty sixth', '27th': 'twenty seventh',
    '28th': 'twenty eighth', '29th': 'twenty ninth', '30th': 'thirtieth',
    '31st': 'thirty first',
}

def _expand_ordinals(text):
    """Expand ordinal numbers (1st → first, etc.)."""
    def _ordinal_match(m):
        key = m.group(0).lower()
        return _ordinal_map.get(key, m.group(0))
    return re.sub(r'\b\d{1,2}(?:st|nd|rd|th)\b', _ordinal_match, text, flags=re.IGNORECASE)


# ==============================================================================
# ABBREVIATIONS
# ==============================================================================

_abbreviations = [
    (re.compile(r'\bMr\.', re.IGNORECASE), 'Mister'),
    (re.compile(r'\bMrs\.', re.IGNORECASE), 'Missis'),
    (re.compile(r'\bMs\.', re.IGNORECASE), 'Miss'),
    (re.compile(r'\bDr\.', re.IGNORECASE), 'Doctor'),
    (re.compile(r'\bProf\.', re.IGNORECASE), 'Professor'),
    (re.compile(r'\bSr\.', re.IGNORECASE), 'Senior'),
    (re.compile(r'\bJr\.', re.IGNORECASE), 'Junior'),
    (re.compile(r'\bSt\.', re.IGNORECASE), 'Saint'),
    (re.compile(r'\bGen\.', re.IGNORECASE), 'General'),
    (re.compile(r'\bGov\.', re.IGNORECASE), 'Governor'),
    (re.compile(r'\bSgt\.', re.IGNORECASE), 'Sergeant'),
    (re.compile(r'\bCpt\.', re.IGNORECASE), 'Captain'),
    (re.compile(r'\bLt\.', re.IGNORECASE), 'Lieutenant'),
    (re.compile(r'\bRev\.', re.IGNORECASE), 'Reverend'),
    (re.compile(r'\bSen\.', re.IGNORECASE), 'Senator'),
    (re.compile(r'\bRep\.', re.IGNORECASE), 'Representative'),
    (re.compile(r'\betc\.', re.IGNORECASE), 'et cetera'),
    (re.compile(r'\bvs\.', re.IGNORECASE), 'versus'),
    (re.compile(r'\bJan\.', re.IGNORECASE), 'January'),
    (re.compile(r'\bFeb\.', re.IGNORECASE), 'February'),
    (re.compile(r'\bMar\.', re.IGNORECASE), 'March'),
    (re.compile(r'\bApr\.', re.IGNORECASE), 'April'),
    (re.compile(r'\bJun\.', re.IGNORECASE), 'June'),
    (re.compile(r'\bJul\.', re.IGNORECASE), 'July'),
    (re.compile(r'\bAug\.', re.IGNORECASE), 'August'),
    (re.compile(r'\bSep\.', re.IGNORECASE), 'September'),
    (re.compile(r'\bSept\.', re.IGNORECASE), 'September'),
    (re.compile(r'\bOct\.', re.IGNORECASE), 'October'),
    (re.compile(r'\bNov\.', re.IGNORECASE), 'November'),
    (re.compile(r'\bDec\.', re.IGNORECASE), 'December'),
    (re.compile(r'\be\.g\.', re.IGNORECASE), 'for example'),
    (re.compile(r'\bi\.e\.', re.IGNORECASE), 'that is'),
    (re.compile(r'\bno\.', re.IGNORECASE), 'number'),
    (re.compile(r'\bvol\.', re.IGNORECASE), 'volume'),
    (re.compile(r'\bft\.', re.IGNORECASE), 'feet'),
    (re.compile(r'\bin\.', re.IGNORECASE), 'inches'),
    (re.compile(r'\blbs?\.', re.IGNORECASE), 'pounds'),
    (re.compile(r'\boz\.', re.IGNORECASE), 'ounces'),
    (re.compile(r'\bmi\.', re.IGNORECASE), 'miles'),
    (re.compile(r'\bkm\.', re.IGNORECASE), 'kilometers'),
]

def _expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = regex.sub(replacement, text)
    return text


# ==============================================================================
# TIME
# ==============================================================================

def _expand_time(text):
    """Expand time expressions (3:15 → three fifteen)."""
    def _time_match(m):
        hour = int(m.group(1))
        minute = int(m.group(2))
        if minute == 0:
            return _int_to_words(hour) + " o'clock"
        return _int_to_words(hour) + ' ' + _int_to_words(minute)
    return re.sub(r'\b(\d{1,2}):(\d{2})\b', _time_match, text)


# ==============================================================================
# YEARS
# ==============================================================================

def _expand_years(text):
    """Expand 4-digit years (2024 → twenty twenty four, 1985 → nineteen eighty five)."""
    def _year_match(m):
        year = int(m.group(0))
        if 1000 <= year <= 2099:
            first_half = year // 100
            second_half = year % 100
            if second_half == 0:
                return _int_to_words(first_half) + ' hundred'
            elif second_half < 10:
                return _int_to_words(first_half) + ' oh ' + _int_to_words(second_half)
            else:
                return _int_to_words(first_half) + ' ' + _int_to_words(second_half)
        return _int_to_words(year)
    # Match 4-digit numbers that look like years (preceded by space/start, not part of larger number)
    return re.sub(r'(?<!\d)\b(1[0-9]{3}|20[0-9]{2})\b(?!\d)', _year_match, text)


# ==============================================================================
# UNICODE & WHITESPACE
# ==============================================================================

def _normalize_unicode(text):
    """Normalize unicode characters to ASCII equivalents."""
    # Common replacements
    replacements = {
        '\u2018': "'", '\u2019': "'",  # smart quotes
        '\u201c': '"', '\u201d': '"',
        '\u2014': '--', '\u2013': '-',  # em/en dash
        '\u2026': '...',               # ellipsis
        '\u00a0': ' ',                 # non-breaking space
        '\u00ad': '',                  # soft hyphen
        '\u200b': '',                  # zero-width space
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Normalize remaining unicode
    text = unicodedata.normalize('NFKD', text)
    return text


def _collapse_whitespace(text):
    """Collapse multiple spaces/newlines into single space."""
    return re.sub(r'\s+', ' ', text).strip()


# ==============================================================================
# SYMBOLS
# ==============================================================================

def _expand_symbols(text):
    """Expand common symbols to words."""
    text = text.replace('%', ' percent')
    text = text.replace('&', ' and ')
    text = text.replace('+', ' plus ')
    text = text.replace('=', ' equals ')
    text = text.replace('@', ' at ')
    text = text.replace('#', ' number ')
    return text


_letter_to_word = {
    'a': 'ay', 'b': 'bee', 'c': 'see', 'd': 'dee', 'e': 'ee', 'f': 'ef', 'g': 'jee',
    'h': 'aitch', 'i': 'eye', 'j': 'jay', 'k': 'kay', 'l': 'el', 'm': 'em', 'n': 'en',
    'o': 'oh', 'p': 'pee', 'q': 'cue', 'r': 'are', 's': 'ess', 't': 'tee', 'u': 'you',
    'v': 'vee', 'w': 'double u', 'x': 'ex', 'y': 'wye', 'z': 'zee'
}

def _expand_acronyms(text):
    """
    Expand acronyms into their spoken letter names to provide enough character slots
    for phonetic alignment. E.g., "RTX" -> "are tee ex", "AI" -> "ay eye"
    """
    def _replace_acronym(m):
        letters = m.group(1)
        suffix = m.group(2) or "" # Handle plural 's' like in PCs
        expanded = ' '.join([_letter_to_word.get(l.lower(), l) for l in letters])
        if suffix.lower() == 's':
            expanded += 's'
        return expanded

    # Matches uppercase sequences of 2+ letters, optionally followed by a lowercase 's'
    return re.sub(r'\b([A-Z]{2,})(s?)\b', _replace_acronym, text)


# ==============================================================================
# MAIN NORMALIZER
# ==============================================================================

def normalize_text(text):
    """
    Full Tacotron2-style text normalization pipeline.
    
    Converts written text into a spoken-form representation suitable
    for character-level TTS models.
    
    Args:
        text: Raw input text string
        
    Returns:
        Normalized text string (lowercase, numbers expanded, abbreviations expanded)
    """
    text = _normalize_unicode(text)
    text = _expand_abbreviations(text)
    text = _expand_currency(text)
    text = _expand_time(text)
    text = _expand_ordinals(text)
    text = _expand_years(text)
    text = _expand_symbols(text)
    text = _expand_acronyms(text)
    # Numbers last (after currency/time/ordinals/years have been handled)
    text = re.sub(r'\d[\d,]*\.?\d*', _number_to_words, text)
    text = _collapse_whitespace(text)
    text = text.lower()
    return text


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    tests = [
        "Mr. Smith paid $3.50 for 2 items on Jan. 1st, 2024.",
        "The temperature is 72.5 degrees at 3:15 PM.",
        "Dr. Johnson lives at 123 Main St., Apt. 4B.",
        "She earned $1,000,000 in 1985, i.e., she was very successful.",
        "The 3rd item costs £25 and weighs 10 lbs.",
        "It's 100% certain that 2+2=4.",
        "The train departs at 14:30 from platform 9.",
        "He ran 26.2 mi. in the 42nd annual marathon.",
        "Gov. Smith & Sen. Jones met on Feb. 14th, 2023.",
        "The U.S. population is approx. 330,000,000.",
    ]
    
    for text in tests:
        normalized = normalize_text(text)
        print(f"  IN:  {text}")
        print(f"  OUT: {normalized}")
        print()
