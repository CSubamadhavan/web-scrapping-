
import re

def extract_contact_info(content):
    contact_info = {}

    # Detect all emails in text
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", content)
    if emails:
        contact_info["mail"] = emails

    # Detect all Indian phone numbers (+91 10 digits OR 10 digits starting with 6-9)
    phones = re.findall(r"(?:\+91[\-\s]?)?\d{5}[\s]?\d{5}", content)
    if phones:
        contact_info["phone"] = phones

    # Detect address by looking for known keywords and pattern till city & pincode
    address_match = re.search(
        r"(?:Door\s*No|No\.?)\s*\d+[\w\s,.-]+?(?:Road|Street|St|Colony|Nagar|Lane|Avenue|Byepass)[\w\s,.-]+?\d{6}",
        content,
        re.IGNORECASE
    )
    if address_match:
        contact_info["address"] = address_match.group().strip()

    return contact_info
