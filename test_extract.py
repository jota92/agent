#!/usr/bin/env python3
import re
from typing import Optional

def extract_search_query(instruction: str) -> Optional[str]:
    """Try to infer the keyword the user wants to search for."""
    if not instruction:
        return None
    
    # Match "googleでXを検索" - capturing X
    google_pattern = re.search(r'googleで(.+?)(?:を?検索|を?調べて)', instruction, re.IGNORECASE)
    if google_pattern:
        candidate = google_pattern.group(1).strip(" 。、,.!?\"'」』 ")
        print(f"  Matched google_pattern: '{candidate}'")
        if candidate:
            return candidate
    
    # Fallback: match anything before を検索
    fallback = re.search(r'(.+?)を?検索', instruction, re.IGNORECASE)
    if fallback:
        candidate = fallback.group(1).strip(" 。、,.!?\"'」』googleで ")
        print(f"  Matched fallback: '{candidate}'")
        if candidate and candidate not in ["google", "グーグル"]:
            return candidate
    
    return None

# Test cases
tests = [
    "googleでopenaiを検索して一番上の結果をクリックして",
    "pythonを検索",
    "googleでAIについて調べて"
]

for test in tests:
    result = extract_search_query(test)
    print(f"Input:  '{test}'")
    print(f"Result: '{result}'")
    print()
