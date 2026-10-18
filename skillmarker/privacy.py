"""Remove contact details before text leaves your machine.

Job ads usually end with a recruiter's name, email and phone number. They are useless
for skill analysis, cost tokens, and are personal data, so they never go to the LLM.
The same applies to a CV pasted into the app.
"""
import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?:\+|00)?\d[\d\s/()-]{7,}\d")
URL = re.compile(r"https?://\S+|www\.\S+")
CONTACT_LINE = re.compile(r"^(?:ansprechpartner(?:in)?|kontakt|contact(?: person)?|ihr(?:e)? ansprechpartner(?:in)?)\b.*$",
                          re.IGNORECASE | re.MULTILINE)
SALUTATION_NAME = re.compile(r"\b(?:Frau|Herr|Mr\.?|Mrs\.?|Ms\.?)\s+[A-ZÄÖÜ][\w-]+(?:\s+[A-ZÄÖÜ][\w-]+)?")


def scrub(text: str) -> str:
    if not text:
        return ""
    text = CONTACT_LINE.sub("[contact removed]", text)
    text = EMAIL.sub("[email]", text)
    text = URL.sub("[link]", text)
    text = PHONE.sub("[phone]", text)
    return SALUTATION_NAME.sub("[name]", text)
