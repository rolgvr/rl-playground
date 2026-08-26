"""Repair app.js mojibake from a UTF-8 -> (Windows-1252 misread) -> UTF-8 trip.

.NET's Windows-1252 decoder maps the 5 'undefined' bytes (0x81,0x8D,0x8F,0x90,
0x9D) to the matching C1 control codepoints rather than failing, so we rebuild
that exact byte<->char map to reverse it precisely.
"""
import sys

path = sys.argv[1]
text = open(path, "rb").read().decode("utf-8-sig")   # strip BOM, read as UTF-8

# forward map: byte -> char, exactly as .NET's Windows-1252 decoder produces
forward = {}
for b in range(256):
    try:
        forward[b] = bytes([b]).decode("cp1252")
    except UnicodeDecodeError:
        forward[b] = chr(b)            # undefined byte -> C1 control (identity)
reverse = {ch: b for b, ch in forward.items()}

out = bytearray()
bad = 0
for ch in text:
    if ch in reverse:
        out.append(reverse[ch])
    else:
        out.extend(ch.encode("utf-8"))  # genuine char (shouldn't happen)
        bad += 1

fixed = bytes(out).decode("utf-8").replace("\r\n", "\n")
open(path, "w", encoding="utf-8", newline="\n").write(fixed)
print(f"repaired {path}: {len(fixed)} chars, {bad} unmapped")
for tok in ["★ least work", "— the two points", "▶ Race", "🧊", "Bi-A*"]:
    print(f"  contains {tok!r}: {tok in fixed}")
