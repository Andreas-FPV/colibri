"""Summarise a GGUF's tensor types (reads only the header; works on a partial download).

usage: python gguf_types.py FILE.gguf
"""
import struct
import sys
from collections import Counter, defaultdict

GGML = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
        10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
        17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
        23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
        30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4"}


def main(path):
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    def rstr():
        return f.read(rd("Q")).decode("utf-8", "replace")
    def rval(t):
        sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if t in sizes:
            return rd(sizes[t])
        if t == 8:
            return rstr()
        if t == 9:
            et, n = rd("I"), rd("Q")
            vals = [rval(et) for _ in range(n)]
            return vals if n <= 8 else f"[{n} x type{et}]"
        raise ValueError(t)
    assert f.read(4) == b"GGUF", "not a GGUF"
    ver, nt, nkv = rd("I"), rd("Q"), rd("Q")
    print(f"GGUF v{ver}: {nt} tensors, {nkv} metadata keys")
    for _ in range(nkv):
        k = rstr(); v = rval(rd("I"))
        if any(s in k for s in ("architecture", "block_count", "expert", "loop", "file_type", "feed_forward", "embedding_length", "quantize")):
            print(f"  {k} = {v}")
    by_kind = defaultdict(Counter)
    for _ in range(nt):
        name = rstr(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; t = rd("I"); rd("Q")
        kind = name.split(".", 2)[-1] if name.startswith("blk.") else name
        by_kind[kind][GGML.get(t, f"type{t}")] += 1
        if name.startswith("blk.0.ffn") or name.startswith("blk.6.ffn"):
            print(f"  {name:40s} {GGML.get(t, t):8s} {dims}")
    print("\ntensor kinds and their types (count over layers):")
    for kind in sorted(by_kind):
        print(f"  {kind:40s} {dict(by_kind[kind])}")


if __name__ == "__main__":
    main(sys.argv[1])
