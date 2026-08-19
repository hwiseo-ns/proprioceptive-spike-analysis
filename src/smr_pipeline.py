#!/usr/bin/env python3
"""
Rabbit PNS spike sorting — Ch7 working channel, video001.SMR
============================================================
Complete pipeline as validated so far. Only surviving methods are included;
approaches that failed a null test or a ground-truth check have been removed.

STAGES
  1. read + filter        Ch1-7, zero-phase notch/bandpass/Savitzky-Golay
  2. detect               global fixed threshold on Ch7
  3. P/N resort           assign each event a polarity family by whole-window match
  4. genuineness filter   half-width bounds only (no r-gating: r is the clustering
                          variable and gating on it upstream would suppress a
                          minority unit before clustering could see it)
  5. burst segmentation   temporal proximity only, no waveform information
  6. clustering (v8)      per burst x polarity family, recursive peel

KEY PARAMETERS (all explicit; none hidden)
  fs 30000 Hz          gain Ch7 0.00486374 uV/bit      duration 511.06 s
  notch                60/120/180 Hz only, Q=20, iirnotch -> tf2sos -> sosfiltfilt
  bandpass             150-3000 Hz, Butterworth order 4, sosfiltfilt
  smoothing            Savitzky-Golay, window 25, polyorder 7
  CAR/CMR, Wiener      NOT applied
  MAD sigma            median|y|/0.6745                        = 2.769 uV
  true sigma           MAD of the 1500-2900 Hz band x 1.981    = 2.643 uV
  threshold            4.5 x MAD sigma = 12.46 uV (= 4.71 x true sigma)
  dead time            1.5 ms      alignment  x4 upsample, +/-0.5 ms
  waveform window      -2.0/+3.0 ms  => 150 samples (PRE 60, POST 90)
  P/N pairing window   +/-3 ms
  NOISESHAPE reject    half-width < 0.30 ms or > 2.5 ms   (width-only)
  COLLISION flag       2nd supra-threshold crossing 1.5-3 ms away; kept for rate AND
                       eligible for clustering, but excluded from template estimation
  burst rule           gap <= min(200 ms, max(1.5 x running-median ISI, 20 ms)) x 2
  burst minimum        6 spikes
  clustering R_HIGH    0.94   (core formation; = pooled within-cluster r measured)
  clustering R_LO      0.80   (membership)
  amplitude window     amp >= mode x 10^-0.13   -- ONE-SIDED, no ceiling
                       mode = largest mode of log10(amp) among r>=R_HIGH spikes,
                       Gaussian mean-shift, bandwidth 0.06 dex
  refractory           3.0 ms, strongest-first admission
  periodicity          CONDITIONAL: applied only if the member set beats a
                       uniform-within-span null by >= 0.15; then members with local
                       phase residual > 0.25 are dropped (reverted if that would
                       break MIN_RUN)
  MIN_RUN              5        max peel passes 5

Usage
  python smr_pipeline.py --smr /path/video001.SMR --out ./work
  (writes work/clusters.npz and work/profiles.npz; re-run is cached)
"""
import argparse, os, sys
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos, savgol_filter
from scipy.interpolate import CubicSpline

FS = 30000.0
PRE, POST = 60, 90
THRESH_K = 4.5
DEAD_MS, SEARCH_MS, UPSAMPLE = 1.5, 0.5, 4
PAIR_MS = 3.0
HW_LO, HW_HI = 0.30, 2.5
COLL_LO_MS, COLL_HI_MS = 1.5, 3.0
BURST_CEIL, BURST_MULT, BURST_FLOOR, MAX_SKIP = 0.200, 1.5, 0.020, 1
BURST_MIN = 6
R_HIGH, R_LO, K_LOG = 0.94, 0.80, 0.13
H_LOG = 0.06
T_REF = 0.003
PERIODIC_MARGIN, PHASE_TOL, N_NULL = 0.15, 0.25, 20
MIN_RUN, MAX_PASS = 5, 5
HIGHBAND = (1500., 2900.)
HB_CALIB = 1.981

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DEFAULT_SMR = os.path.join(ROOT, 'data', 'video001.SMR')
DEFAULT_OUT = os.path.join(ROOT, 'results')


# ---------------------------------------------------------------- 1. read/filter
def read_channels(smr_path, channels=(1, 2, 3, 4, 5, 6, 7)):
    from neo.rawio import Spike2RawIO
    r = Spike2RawIO(filename=smr_path, try_signal_grouping=False)
    r.parse_header()
    sig = r.header['signal_channels']
    n = int(r.get_signal_size(0, 0, stream_index=channels[0] - 1))
    out = np.empty((len(channels), n), np.float32)
    for j, c in enumerate(channels):
        g = float(sig[c - 1]['gain'])
        buf = np.empty(n, np.float32)
        for a in range(0, n, 2_000_000):
            b = min(a + 2_000_000, n)
            buf[a:b] = r.get_analogsignal_chunk(0, 0, a, b, stream_index=c - 1).ravel() * g
        out[j] = buf
    return out


def filter_chain(x, fs=FS):
    y = np.asarray(x, np.float64)
    for f0 in (60., 120., 180.):
        b, a = iirnotch(f0, 20., fs)
        y = sosfiltfilt(tf2sos(b, a), y)
    y = sosfiltfilt(butter(4, [150., 3000.], btype='band', fs=fs, output='sos'), y)
    return savgol_filter(y, 25, 7)


def mad_sigma(y):
    return float(np.median(np.abs(y)) / 0.6745)


def true_sigma(y, fs=FS):
    yh = sosfiltfilt(butter(4, list(HIGHBAND), btype='band', fs=fs, output='sos'), y)
    return mad_sigma(yh) * HB_CALIB


# ------------------------------------------------------------------- 2. detect
def detect(y, thr, polarity, fs=FS):
    m = (y < -thr) if polarity < 0 else (y > thr)
    starts = np.where(np.diff(m.astype(np.int8)) == 1)[0] + 1
    dead = int(DEAD_MS * 1e-3 * fs)
    keep, last = [], -10**9
    for i in starts:
        if i - last > dead:
            keep.append(i); last = i
    idx = np.asarray(keep, np.int64)
    g = int(SEARCH_MS * 1e-3 * fs)
    idx = idx[(idx > g + 4) & (idx < len(y) - g - 4)]
    tt = np.arange(-g, g + 1)
    tf = np.arange(-g, g, 1.0 / UPSAMPLE)
    al = np.empty(len(idx), np.int64)
    for k, i in enumerate(idx):
        fine = CubicSpline(tt, y[i - g:i + g + 1])(tf)
        al[k] = i + int(round(tf[np.argmin(fine) if polarity < 0 else np.argmax(fine)]))
    al = np.unique(al)
    keep, last = [], -10**9
    for i in al:
        if i - last > dead:
            keep.append(i); last = i
    return np.asarray(keep, np.int64)


# -------------------------------------------------------------- 3. P/N resort
def _norm_rows(W):
    Xc = W - W.mean(1, keepdims=True)
    return Xc / np.clip(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12, None)


def resort_polarity(y, neg, pos, fs=FS):
    """Canonicalise every event on its window trough, then decide polarity family by
    whole-window template match (EM, 3 iterations). Returns event index (trough),
    peak index, and a boolean 'is POS-family'."""
    g = int(0.003 * fs)
    ev = np.concatenate([neg, pos])
    ev = ev[(ev > g + PRE + 2) & (ev < len(y) - g - POST - 2)]
    tr = np.empty(len(ev), np.int64); pk = np.empty(len(ev), np.int64)
    for k, i in enumerate(ev):
        w = y[i - g:i + g + 1]
        tr[k] = i - g + int(np.argmin(w)); pk[k] = i - g + int(np.argmax(w))
    o = np.argsort(tr); tr, pk = tr[o], pk[o]
    keep = np.concatenate([[True], np.diff(tr) > int(0.0015 * fs)])
    tr, pk = tr[keep], pk[keep]
    Wt = _norm_rows(np.stack([y[i - PRE:i + POST] for i in tr]))
    Wp = _norm_rows(np.stack([y[i - PRE:i + POST] for i in pk]))
    fam = np.abs(y[pk]) > np.abs(y[tr])
    for _ in range(3):
        Tn = np.median(Wt[~fam], 0); Tn /= np.linalg.norm(Tn)
        Tp = np.median(Wp[fam], 0);  Tp /= np.linalg.norm(Tp)
        fam = (Wp @ Tp) > (Wt @ Tn)
    return tr, pk, fam


# --------------------------------------------------------- 4. genuineness filter
def half_width(w, polarity=-1):
    v = w if polarity < 0 else -w
    half = v[PRE] / 2.0
    a = PRE
    while a > 0 and v[a - 1] < half: a -= 1
    b = PRE
    while b < len(v) - 1 and v[b + 1] < half: b += 1
    return (b - a + 1) / FS * 1e3


def genuineness(y, ev, pk, fam, thr):
    """Width-only rejection + collision flag. Returns verdict array."""
    verdict = np.empty(len(ev), 'U10')
    a1, a2 = int(COLL_LO_MS * 1e-3 * FS), int(COLL_HI_MS * 1e-3 * FS)
    for k in range(len(ev)):
        c = pk[k] if fam[k] else ev[k]
        w = y[c - PRE:c + POST]
        hw = half_width(w, +1 if fam[k] else -1)
        if hw < HW_LO or hw > HW_HI:
            verdict[k] = 'NOISESHAPE'; continue
        i = ev[k]
        coll = (y[i - a2:i - a1] < -thr).any() or (y[i + a1:i + a2] < -thr).any()
        verdict[k] = 'COLLISION' if coll else 'CLEAN'
    return verdict


# ------------------------------------------------------- 5. burst segmentation
def segment(t):
    bursts, cur = [], [t[0]]
    for x in t[1:]:
        gap = x - cur[-1]
        tol = min(BURST_CEIL, max(BURST_MULT * np.median(np.diff(cur)), BURST_FLOOR)) \
              if len(cur) >= 2 else BURST_CEIL
        if gap <= tol * (MAX_SKIP + 1):
            cur.append(x)
        else:
            bursts.append(np.array(cur)); cur = [x]
    bursts.append(np.array(cur))
    return [b for b in bursts if len(b) >= BURST_MIN]


# ------------------------------------------------------------- 6. clustering v8
def largest_mode(a, h=H_LOG):
    x = np.log10(np.clip(a, 1e-9, None))
    modes = []
    for x0 in x:
        m = x0
        for _ in range(50):
            w = np.exp(-0.5 * ((x - m) / h) ** 2)
            m2 = float((w * x).sum() / max(w.sum(), 1e-12))
            if abs(m2 - m) < 1e-6: break
            m = m2
        modes.append(m)
    best, bn = modes[0], -1
    for m in np.unique(np.round(modes, 3)):
        n = int(np.sum(np.abs(x - m) <= h))
        if n > bn: bn, best = n, m
    return best


def _periodic_fit(times, P):
    if len(times) < 3 or P <= 0: return 0.0
    g = np.diff(np.sort(times))
    n = np.maximum(1, np.round(g / P))
    return float(np.mean(np.abs(g - n * P) / (n * P) < PHASE_TOL))


def _is_periodic(times, rng):
    if len(times) < MIN_RUN: return False
    P = float(np.median(np.diff(np.sort(times))))
    fit = _periodic_fit(times, P)
    nulls = []
    for _ in range(N_NULL):
        s = np.sort(rng.uniform(times.min(), times.max(), len(times)))
        nulls.append(_periodic_fit(s, float(np.median(np.diff(s))) if len(s) > 1 else 0))
    return (fit - float(np.mean(nulls))) >= PERIODIC_MARGIN


def _phase_residual(t, i, k=3):
    lo, hi = max(0, i - k), min(len(t), i + k + 1)
    isis = np.diff(t[lo:hi])
    if len(isis) < 2: return 0.0
    P = np.median(isis)
    if P <= 0: return 0.0
    out = []
    for g in ((t[i] - t[i - 1]) if i > 0 else None,
              (t[i + 1] - t[i]) if i < len(t) - 1 else None):
        if g is None: continue
        n = max(1, round(g / P)); out.append(abs(g - n * P) / (n * P))
    return min(out) if out else 0.0


def _pass(t, W, amp, coll, rng):
    Xn = _norm_rows(W)
    src = ~coll if (~coll).sum() >= 3 else np.ones(len(coll), bool)
    T = np.median(Xn[src], 0); T /= np.linalg.norm(T)
    r = Xn @ T
    hi = (r >= R_HIGH) & src
    if hi.sum() < 3:
        return np.zeros(len(t), bool), r, None
    mode_log = largest_mode(amp[hi]); mode_amp = 10 ** mode_log
    la = mode_amp * 10 ** (-K_LOG)                       # one-sided: no ceiling
    cand = (r >= R_LO) & (amp >= la)
    score = r - np.clip(mode_log - np.log10(np.clip(amp, 1e-9, None)), 0, None)
    mem = np.zeros(len(t), bool)
    for i in np.argsort(-score):
        if not cand[i]: continue
        if mem.any() and np.min(np.abs(t[mem] - t[i])) < T_REF: continue
        mem[i] = True
    if mem.sum() >= MIN_RUN and _is_periodic(t[mem], rng):
        idx = np.where(mem)[0]; o = np.argsort(t[idx]); ts = t[idx][o]
        keep = np.array([_phase_residual(ts, j) <= PHASE_TOL for j in range(len(ts))])
        if keep.sum() >= MIN_RUN:
            mem[idx[o][~keep]] = False
    return mem, r, (la, mode_amp)


def cluster(y, ev, fam, coll, seed=0):
    t = ev / FS
    amp = np.abs(y[ev])
    bursts = segment(t)
    be = np.array([[b[0], b[-1]] for b in bursts])
    rng = np.random.default_rng(seed)
    out = np.full(len(t), -1, int); rank = np.full(len(t), -1, int)
    rr = np.full(len(t), np.nan)
    nc, info = 0, []
    for bi, (a, b) in enumerate(be):
        i0, i1 = np.searchsorted(t, a), np.searchsorted(t, b, side='right')
        for fv in (False, True):
            gi = i0 + np.where(fam[i0:i1] == fv)[0]
            if len(gi) < MIN_RUN:
                out[gi] = -2; continue
            W = np.stack([y[j - PRE:j + POST] for j in ev[gi]])
            pool, k, done = np.arange(len(gi)), 0, np.zeros(len(gi), bool)
            while len(pool) >= MIN_RUN and k < MAX_PASS:
                mem, r, win = _pass(t[gi[pool]], W[pool], amp[gi[pool]], coll[gi[pool]], rng)
                rr[gi[pool]] = np.where(np.isnan(rr[gi[pool]]), r, rr[gi[pool]])
                if win is None or mem.sum() < MIN_RUN: break
                out[gi[pool[mem]]] = nc; rank[gi[pool[mem]]] = k; done[pool[mem]] = True
                info.append((bi, int(fv), nc, k, int(mem.sum()),
                             float(np.median(amp[gi[pool][mem]])),
                             float(np.median(t[gi[pool][mem]]))))
                nc += 1; pool = pool[~mem]; k += 1
            out[gi[~done]] = np.where(out[gi[~done]] >= 0, out[gi[~done]], -2)
    return dict(t=t, amp=amp, out=out, rank=rank, r=rr, n_clusters=nc,
                info=np.array(info, float), burst_edges=be)


# ------------------------------------------------------- multichannel profiles
def profiles(Yall, ev, out, cluster_ids):
    """Median waveform on every channel + per-channel peak-to-peak, per cluster."""
    nch = Yall.shape[0]
    prof = np.zeros((len(cluster_ids), nch, PRE + POST))
    p2p = np.zeros((len(cluster_ids), nch))
    lat = np.zeros((len(cluster_ids), nch))
    for k, c in enumerate(cluster_ids):
        idx = ev[out == c]
        idx = idx[(idx > PRE + 20) & (idx < Yall.shape[1] - POST - 20)]
        if len(idx) == 0: continue
        W = np.stack([Yall[:, i - PRE:i + POST] for i in idx])
        W = W - W.mean(axis=2, keepdims=True)
        M = np.median(W, 0)
        prof[k] = M
        p2p[k] = M.max(1) - M.min(1)
        tr = np.argmin(M, 1)
        lat[k] = (tr - tr[-1]) / FS * 1e3
    return prof, p2p, lat


# --------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smr', default=DEFAULT_SMR)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--work-channel', type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fcache = os.path.join(a.out, 'filtered.npy')

    if os.path.exists(fcache):
        Yall = np.load(fcache, mmap_mode='r')
        print(f'[cache] filtered channels {Yall.shape}')
    else:
        print(f'[1] reading + filtering Ch1-7 from {a.smr} ...')
        raw = read_channels(a.smr)
        Yall = np.stack([filter_chain(raw[j]).astype(np.float32) for j in range(raw.shape[0])])
        np.save(fcache, Yall)
    y = np.asarray(Yall[a.work_channel - 1], np.float64)

    s_mad, s_true = mad_sigma(y), true_sigma(y)
    thr = THRESH_K * s_mad
    print(f'[2] MAD sigma {s_mad:.4f} | true sigma {s_true:.4f} | threshold {thr:.3f} uV '
          f'(= {thr/s_true:.2f} x true sigma)')
    neg, pos = detect(y, thr, -1), detect(y, thr, +1)
    print(f'    NEG {len(neg)}  POS {len(pos)}')

    print('[3] polarity resort ...')
    ev, pk, fam = resort_polarity(y, neg, pos)
    print(f'    events {len(ev)}  NEG-family {np.sum(~fam)}  POS-family {np.sum(fam)}')

    print('[4] genuineness filter ...')
    verdict = genuineness(y, ev, pk, fam, thr)
    print('    ' + '  '.join(f'{v} {int(np.sum(verdict==v))}'
                             for v in ('CLEAN', 'COLLISION', 'NOISESHAPE')))
    keep = verdict != 'NOISESHAPE'
    ev2, fam2, coll2 = ev[keep], fam[keep], verdict[keep] == 'COLLISION'

    print('[5+6] burst segmentation + clustering ...')
    R = cluster(y, ev2, fam2, coll2)
    print(f'    bursts {len(R["burst_edges"])}  clusters {R["n_clusters"]}  '
          f'members {int(np.sum(R["out"]>=0))}  intruders {int(np.sum(R["out"]==-2))}')

    info = R['info']
    peel0_neg = np.array([int(c[2]) for c in info if c[1] == 0 and c[3] == 0], int)
    sizes = np.array([int(c[4]) for c in info if c[1] == 0 and c[3] == 0])
    print(f'    peel-0 NEG clusters {len(peel0_neg)}  (>=20 spikes: {int(np.sum(sizes>=20))})')

    print('[7] multichannel profiles ...')
    allc = np.array(sorted(set(R['out'][R['out'] >= 0].tolist())), int)
    prof, p2p, lat = profiles(np.asarray(Yall), ev2, R['out'], allc)

    np.savez_compressed(os.path.join(a.out, 'clusters.npz'),
                        ev=ev2, fam=fam2, coll=coll2, **R)
    np.savez_compressed(os.path.join(a.out, 'profiles.npz'),
                        cluster_ids=allc, prof=prof, p2p=p2p, lat=lat,
                        peel0_neg=peel0_neg)
    print(f'[done] wrote {a.out}/clusters.npz and {a.out}/profiles.npz')


if __name__ == '__main__':
    main()
