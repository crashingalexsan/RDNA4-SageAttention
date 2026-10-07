"""Make the sage kernel shape-generic for ahead-of-time packaging.

Applied after make_sage_r2.py. Head counts become runtime kernel arguments (appended after Sink),
so one code object per (head_dim, out dtype, arch) serves every model; the output dtype becomes a
build option (bf16 or f16).
"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "kernels/attention/flash_attn_sage_gfx120x.py"
src = path.read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    if n != count:
        raise SystemExit(f"expected {count} match(es), got {n}: {old[:80]!r}")
    src = src.replace(old, new)


sub('''    row_subtiles: int = 2,
) -> Callable[..., None]:
''', '''    row_subtiles: int = 2,
    out_dtype: str = "bf16",
) -> Callable[..., None]:
''')
# The K loader covers BLOCK_N rows in NUM_BATCHES_KV passes of ROWS_PER_BATCH_LOAD rows; when the rows
# per pass don't divide BLOCK_N some K rows are never loaded (e.g. head_dim 96 with 128 threads).
sub('''    else:
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
        KV_NEEDS_GUARD = False
''', '''    else:
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
        KV_NEEDS_GUARD = False
    assert ROWS_PER_BATCH_LOAD >= BLOCK_N or BLOCK_N % ROWS_PER_BATCH_LOAD == 0, (
        f"K loader: {BLOCK_SIZE} threads cover {ROWS_PER_BATCH_LOAD} rows of head_dim {HEAD_DIM} per pass, "
        f"which does not tile BLOCK_N={BLOCK_N}; use another block_m"
    )
''')
sub('''    out_numeric_cls = fx.BFloat16
''', '''    assert out_dtype in ("bf16", "f16"), f"out_dtype must be bf16 or f16, got {out_dtype!r}"
    out_numeric_cls = fx.Float16 if out_dtype == "f16" else fx.BFloat16
''')

# kernel and launcher signatures: runtime head counts after Sink
sub('''        Sink: fx.Pointer,
    ) -> None:
        elem_dtype = elem_numeric_cls
''', '''        Sink: fx.Pointer,
        num_heads_rt: fx.Int32,
        num_kv_heads_rt: fx.Int32,
    ) -> None:
        elem_dtype = elem_numeric_cls
        nh_u = fx.Uint64(num_heads_rt)
        nkv_u = fx.Uint64(num_kv_heads_rt)
        stride_token_u = nh_u * fx.Uint64(HEAD_DIM)
        kv_stride_u = nkv_u * fx.Uint64(HEAD_DIM)
''')
sub('''        Sink: fx.Pointer,
        stream: fx.Stream = fx.Stream(''', '''        Sink: fx.Pointer,
        num_heads_rt: fx.Int32,
        num_kv_heads_rt: fx.Int32,
        stream: fx.Stream = fx.Stream(''')
sub('''            Sink,
        )
''', '''            Sink,
            num_heads_rt,
            num_kv_heads_rt,
        )
''')
sub("        grid_x = bs_idx * num_q_tiles * NUM_HEADS\n", "        grid_x = bs_idx * num_q_tiles * fx.Uint64(num_heads_rt)\n")

# kernel body: compile-time head counts -> runtime values
sub("        head_idx = block_id % NUM_HEADS\n", "        head_idx = block_id % nh_u\n")
sub("        kv_head_idx = head_idx // fx.Uint64(KV_GROUP)\n", "        kv_head_idx = head_idx // (nh_u // nkv_u)\n")
sub("        batch_q_tile_id = block_id // NUM_HEADS\n", "        batch_q_tile_id = block_id // nh_u\n")
sub("            return token * STRIDE_TOKEN + head_idx * HEAD_DIM + col\n", "            return token * stride_token_u + head_idx * HEAD_DIM + col\n")
sub("            return token * KV_STRIDE + kv_head_idx * HEAD_DIM + col\n", "            return token * kv_stride_u + kv_head_idx * HEAD_DIM + col\n")
sub("        v_batch_elems = seq_len_kv_v * fx.Uint64(KV_STRIDE)\n", "        v_batch_elems = seq_len_kv_v * kv_stride_u\n")
sub("            return token_idx * KV_STRIDE + kv_head_idx * HEAD_DIM + col\n", "            return token_idx * kv_stride_u + kv_head_idx * HEAD_DIM + col\n")
sub("        _q_scale_base = (fx.Int64(batch_idx) * fx.Int64(NUM_HEADS) + fx.Int64(head_idx)) * fx.Int64(seq_len)\n",
    "        _q_scale_base = (fx.Int64(batch_idx) * fx.Int64(num_heads_rt) + fx.Int64(head_idx)) * fx.Int64(seq_len)\n")
sub("        _k_scale_row = (fx.Int64(batch_idx) * fx.Int64(NUM_KV_HEADS) + fx.Int64(kv_head_idx)) * (\n",
    "        _k_scale_row = (fx.Int64(batch_idx) * fx.Int64(num_kv_heads_rt) + fx.Int64(kv_head_idx)) * (\n")
sub("        _v_scale_base = (fx.Int64(batch_idx) * fx.Int64(NUM_KV_HEADS) + fx.Int64(kv_head_idx)) * fx.Int64(HEAD_DIM) + klane * 8\n",
    "        _v_scale_base = (fx.Int64(batch_idx) * fx.Int64(num_kv_heads_rt) + fx.Int64(kv_head_idx)) * fx.Int64(HEAD_DIM) + klane * 8\n")

path.write_text(src)
print("wrote", path)
