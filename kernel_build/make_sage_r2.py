"""Rewrite the sage kernel main body so each wave owns ROW_SUBTILES x 16 query rows.

Input: kernels/attention/flash_attn_sage_gfx120x.py produced by make_sage.py.
K and V operands loaded from LDS feed ROW_SUBTILES WMMAs each. Non-causal or causal
(bottom-right aligned: query i attends keys j <= i + kv_len - q_len); no bias / sink / LSE.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "kernels/attention/flash_attn_sage_gfx120x.py"
src = path.read_text()


def sub(old, new):
    global src
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"expected 1 match, got {n}: {old[:70]!r}")
    src = src.replace(old, new)


sub('''    num_kv_heads: int | None = None,
) -> Callable[..., None]:
''', '''    num_kv_heads: int | None = None,
    row_subtiles: int = 2,
) -> Callable[..., None]:
''')
sub('''    ROWS_PER_WAVE = WMMA_M
''', '''    ROW_SUBTILES = int(row_subtiles)
    assert ROW_SUBTILES in (1, 2)
    ROWS_PER_WAVE = WMMA_M * ROW_SUBTILES
''')
sub('''    assert not (has_attn_bias or has_per_head_bias), "sage gfx120x FA does not support bias"
''', '''    assert not (has_attn_bias or has_per_head_bias), "sage gfx120x FA does not support bias"
    assert not (has_sink or return_lse), "sage gfx120x FA has no sink/LSE outputs"
''')

sub('''    assert BLOCK_N == 32, (
        f"gfx120x FA score-lane masks assume BLOCK_N==32 (got {BLOCK_N}); "
        "generalize N_SUB_TILES masking before using other block_n"
    )
''', '''    assert BLOCK_N in (32, 64), f"sage gfx120x FA supports BLOCK_N 32 or 64 (got {BLOCK_N})"
''')

# V is quantized per (batch, kv head, channel): VDescale is the channel amax [B, Hkv, D] fp32, V_q = V * 448 / amax
# and O[:, c] = (P @ V_q)[:, c] * amax[c] / 448, applied once in the epilogue.
sub('''        v_descale = _load_descale(VDescale)
''', '''        v_scale_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, VDescale.address_space), VDescale)
''')


start = src.index("        q_row = q_start + wave_q_offset + lane16\n")
end = src.index("    @flyc.jit\n    def launch_flash_attn_sage_func(")
body = '''        R = ROW_SUBTILES
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)
        c_zero_v8i32 = Vec.filled(8, 0, fx.Int32)
        c_rescale_thr = fx.Float32(8.0)
        width_i32 = fx.Int32(WARP_SIZE)
        shuf_16_i32 = fx.Int32(16)

        def reduction_peer(v_f32: fx.Float32 | fx.Vector) -> fx.Float32:
            return fx.gpu.shuffle_xor(fx.Float32(v_f32), shuf_16_i32, width_i32)

        # Per row sub-tile: this lane's query row, its int8 Q fragments and Q scale.
        q_rows = []
        q_in_bounds = []
        q_b_packs = []
        c_logit_scale = []
        _q_scale_base = (fx.Int64(batch_idx) * fx.Int64(NUM_HEADS) + fx.Int64(head_idx)) * fx.Int64(seq_len)
        for sub in range_constexpr(R):
            q_row = q_start + wave_q_offset + fx.Int64(sub * WMMA_M) + lane16
            inb = q_row < seq_len_v
            q_row_safe = fx.Int64(inb.select(q_row, fx.Int64(0)))
            packs = []
            for ks in range_constexpr(K_STEPS_QK):
                q_col = fx.Int64(ks * K_STEP_QK) + klane * WMMA_LANE_K
                packs.append(load_global_v8f8(q_elem_ptr, global_idx(q_row_safe, q_col)))
            q_scale = fx.Float32(fx.ptr_load(q_scale_ptr + fx.Int32(_q_scale_base + q_row_safe)))
            q_rows.append(q_row)
            q_in_bounds.append(inb)
            q_b_packs.append(packs)
            c_logit_scale.append(fx.Float32(sm_scale * _LOG2E) * q_scale)

        _k_scale_row = (fx.Int64(batch_idx) * fx.Int64(NUM_KV_HEADS) + fx.Int64(kv_head_idx)) * (
            fx.Int64(seq_len_kv) // fx.Int64(BLOCK_N)
        )

        # Causal: query row i attends keys j <= i + causal_off (bottom-right alignment).
        causal_off_i32 = seq_len_kv_valid_i32 - fx.Int32(seq_len)
        if const_expr(CAUSAL):
            # a tile needs the element mask once it reaches past this workgroup's first row's last key
            _diag = fx.Int32(q_start) + causal_off_i32 + fx.Int32(1)
            mask_from_i32 = (_diag < seq_len_kv_valid_i32).select(_diag, seq_len_kv_valid_i32)
            # keys past the workgroup's last row's last key are never needed
            _kv_end = fx.Int64(q_start + BLOCK_M) + fx.Int64(causal_off_i32)
            _kv_end = fx.Int64((_kv_end < fx.Int64(0)).select(fx.Int64(0), _kv_end))
            kv_upper = fx.Int64((_kv_end < seq_len_kv_v).select(_kv_end, seq_len_kv_v))
        else:
            mask_from_i32 = seq_len_kv_valid_i32
            kv_upper = seq_len_kv_v

        def _pad_mask(s_vals: list, kv_start_i32: fx.Int32, needs, q_row) -> list:
            if needs:
                klane_off_i32 = fx.Int32(klane) * fx.Int32(8)
                last_key = fx.Int32(q_row) + causal_off_i32
                masked = []
                for si in range_constexpr(NUM_S_VALS):
                    col = kv_start_i32 + fx.Int32((si // 16) * 32 + ((si // 8) % 2) * 16 + si % 8) + klane_off_i32
                    s_m = (col >= seq_len_kv_valid_i32).select(c_neg_inf, s_vals[si])
                    if const_expr(CAUSAL):
                        s_m = (col > last_key).select(c_neg_inf, s_m)
                    masked.append(s_m)
                s_vals = masked
            return s_vals

        def _rescale(o_list: list, need, corr) -> list:
            if need:
                corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(8)
                o_new = []
                for dc in range_constexpr(D_CHUNKS):
                    o_new.append(_fmul(o_list[dc], corr_vec))
                o_list = o_new
            return o_list

        _v_vecs_init = coop_load_v_global(fx.Int64(0))
        init_args = []
        for sub in range_constexpr(R):
            init_args.append(c_neg_inf)
        for sub in range_constexpr(R):
            init_args.append(c_zero_f)
        for _ in range_constexpr(R * D_CHUNKS):
            init_args.append(c_zero_v8f32)
        _V_ARG_BASE = len(init_args)
        for vi in range_constexpr(NUM_V_VECS):
            init_args.append(_v_vecs_init[vi])

        loop_results = init_args
        for kv_block_start, inner_iter_args in range(fx.Int64(0), kv_upper, fx.Int64(BLOCK_N_OUT), init=init_args):
            m_running = [inner_iter_args[sub] for sub in range_constexpr(R)]
            l_running = [inner_iter_args[R + sub] for sub in range_constexpr(R)]
            o_accs = [[inner_iter_args[2 * R + sub * D_CHUNKS + dc] for dc in range_constexpr(D_CHUNKS)] for sub in range_constexpr(R)]
            _v_vecs_tile = [inner_iter_args[_V_ARG_BASE + vi] for vi in range_constexpr(NUM_V_VECS)]

            coop_load_k(kv_block_start, 0)
            gpu.barrier()
            k_base = k_buf_base(0)

            # S = K @ Q^T in i32; each K fragment feeds all row sub-tiles.
            s_accs = [[c_zero_v8i32 for _ in range(NUM_S_ACCS)] for _ in range(R)]
            k_scale = fx.Float32(
                fx.ptr_load(k_scale_ptr + fx.Int32(_k_scale_row + fx.Int64(kv_block_start) // fx.Int64(BLOCK_N)))
            )
            for ks in range_constexpr(K_STEPS_QK):
                k_col = fx.Int64(ks * K_STEP_QK) + klane * WMMA_LANE_K
                for st_idx in range_constexpr(N_SUB_TILES):
                    st_base_row = st_idx * K_SUB_N
                    k_pack_a = Vec(lds_load_i8(k_base + (lane16 + fx.Int64(st_base_row)) * K_STRIDE + k_col, 8))
                    k_pack_b = Vec(lds_load_i8(k_base + (lane16 + fx.Int64(st_base_row + 16)) * K_STRIDE + k_col, 8))
                    for sub in range_constexpr(R):
                        s_accs[sub][st_idx * 2] = wmma_acc_i32(k_pack_a, q_b_packs[sub][ks], s_accs[sub][st_idx * 2])
                        s_accs[sub][st_idx * 2 + 1] = wmma_acc_i32(k_pack_b, q_b_packs[sub][ks], s_accs[sub][st_idx * 2 + 1])

            kv_start_i32 = fx.Int32(kv_block_start)
            tile_needs_pad_mask = kv_start_i32 + fx.Int32(BLOCK_N - 1) >= mask_from_i32

            p_packs = []
            m_next = []
            l_next = []
            o_scaled = []
            for sub in range_constexpr(R):
                s_raw = []
                for st in range_constexpr(NUM_S_ACCS):
                    for r in range_constexpr(8):
                        s_raw.append(fx.Float32(Vec(s_accs[sub][st])[r]))
                s_raw = _pad_mask(s_raw, kv_start_i32, tile_needs_pad_mask, q_rows[sub])

                local_max = s_raw[0]
                for r in range_constexpr(NUM_S_VALS - 1):
                    local_max = _fmax(local_max, s_raw[r + 1])
                row_max = _fmul(_fmax(local_max, reduction_peer(local_max)), k_scale)
                # Keep a stale running max until the tile max exceeds it by 2^8 (P <= 256 fits E4M3).
                m_cand = _fmax(m_running[sub], row_max)
                need_rescale = _fmul(_fsub(m_cand, m_running[sub]), c_logit_scale[sub]) > c_rescale_thr
                m_new_raw = need_rescale.select(m_cand, m_running[sub])
                corr = fx.Float32(
                    fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(_fmul(_fsub(m_running[sub], m_new_raw), c_logit_scale[sub])).ir_value())
                )
                neg_scaled_max = _fsub(c_zero_f, _fmul(c_logit_scale[sub], m_new_raw))
                c_tile_scale = _fmul(c_logit_scale[sub], k_scale)

                p_vals = []
                local_sum = c_zero_f
                for r in range_constexpr(NUM_S_VALS):
                    p = fx.Float32(
                        fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(fx.math.fma(s_raw[r], c_tile_scale, neg_scaled_max)).ir_value())
                    )
                    p_vals.append(p)
                    local_sum = _fadd(local_sum, p)
                tile_sum = _fadd(local_sum, reduction_peer(local_sum))
                l_next.append(_fadd(_fmul(corr, l_running[sub]), tile_sum))
                m_next.append(m_new_raw)
                o_scaled.append(_rescale(o_accs[sub], need_rescale, corr))

                packs_sub = []
                for st_idx in range_constexpr(N_SUB_TILES):
                    packs_st = []
                    for pks in range_constexpr(PV_K_STEPS):
                        p_base = (st_idx * 2 + pks) * 8
                        packs_st.append(fp8_pack_v8([p_vals[p_base + j] for j in range(8)]))
                    packs_sub.append(packs_st)
                p_packs.append(packs_sub)

            coop_store_v_lds(_v_vecs_tile, 0)
            gpu.barrier()
            v_base = v_buf_base(0)
            # O += V^T @ P; each V fragment feeds all row sub-tiles.

            def _load_vt(st_kv_base_val: int, pks_val: int, dc_val: int) -> fx.Vector:
                d_pos = fx.Int64(dc_val * D_CHUNK) + lane16
                kv_row0 = fx.Int64(st_kv_base_val + pks_val * PV_K_STEP) + klane * WMMA_LANE_K
                return Vec(lds_load_i8(v_base + d_pos * VT_STRIDE + kv_row0, 8))

            o_tmp = [list(o_scaled[sub]) for sub in range_constexpr(R)]
            cur_v_packs = [_load_vt(st_idx * K_SUB_N, 0, 0) for st_idx in range_constexpr(N_SUB_TILES)]
            for pks in range_constexpr(PV_K_STEPS):
                for dc in range_constexpr(D_CHUNKS):
                    next_dc = dc + 1
                    next_pks = pks
                    if const_expr(next_dc >= D_CHUNKS):
                        next_dc = 0
                        next_pks = pks + 1
                    has_next = const_expr(next_pks < PV_K_STEPS)
                    next_v_packs = []
                    if const_expr(has_next):
                        for st_idx in range_constexpr(N_SUB_TILES):
                            next_v_packs.append(_load_vt(st_idx * K_SUB_N, next_pks, next_dc))
                    for st_idx in range_constexpr(N_SUB_TILES):
                        for sub in range_constexpr(R):
                            o_tmp[sub][dc] = wmma_acc(cur_v_packs[st_idx], p_packs[sub][st_idx][pks], o_tmp[sub][dc])
                    if const_expr(has_next):
                        cur_v_packs = next_v_packs

            _v_vecs_tile = coop_load_v_global(fx.Int64(kv_block_start) + fx.Int64(BLOCK_N_OUT))

            _yield_args = list(m_next) + list(l_next)
            for sub in range_constexpr(R):
                _yield_args = _yield_args + o_tmp[sub]
            for vi in range_constexpr(NUM_V_VECS):
                _yield_args.append(_v_vecs_tile[vi])
            loop_results = yield _yield_args

        # this lane's 8 output columns per D chunk: dc * 16 + klane * 8 + [0, 8)
        _v_scale_base = (fx.Int64(batch_idx) * fx.Int64(NUM_KV_HEADS) + fx.Int64(kv_head_idx)) * fx.Int64(HEAD_DIM) + klane * 8
        v_scales = []
        for dc in range_constexpr(D_CHUNKS):
            view = fx.make_view(v_scale_ptr + fx.Int32(_v_scale_base + fx.Int64(dc * D_CHUNK)), fx.make_layout(8, 1))
            v_scales.append(Vec(view.load()))
        for sub in range_constexpr(R):
            inv_l = fx.Float32(1.0 / 448.0) / loop_results[R + sub]
            inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(8)
            if q_in_bounds[sub]:
                for dc in range_constexpr(D_CHUNKS):
                    o_norm_vec = _fmul(_fmul(loop_results[2 * R + sub * D_CHUNKS + dc], inv_l_vec), v_scales[dc])
                    o_global = global_idx(q_rows[sub], fx.Int64(dc * D_CHUNK) + klane * 8)
                    _store_global_half(o_elem_ptr, o_global, Vec(o_norm_vec).to(out_dtype))

'''
src = src[:start] + body + src[end:]
path.write_text(src)
print("wrote", path)
