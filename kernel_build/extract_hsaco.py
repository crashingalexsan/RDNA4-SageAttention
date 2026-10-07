"""Pull the device code object out of a FlyDSL 20_gpu_module_to_binary.mlir dump."""
import re, sys
txt = open(sys.argv[1], encoding="utf-8", errors="surrogateescape").read()
i = txt.index('bin = "') + len('bin = "')
out = bytearray()
while True:
    c = txt[i]
    if c == '"':
        break
    if c == "\\":
        nxt = txt[i + 1]
        if nxt in "0123456789abcdefABCDEF":
            out.append(int(txt[i + 1:i + 3], 16)); i += 3; continue
        out.append({"n": 10, "t": 9, '"': 34, "\\": 92}[nxt]); i += 2; continue
    out += c.encode("utf-8", "surrogateescape"); i += 1
open(sys.argv[2], "wb").write(out)
print(f"wrote {sys.argv[2]}: {len(out)} bytes, magic {bytes(out[:4])!r}")
