"""
main.py -- Spike sorting pipeline for rabbit PNS multi-channel recordings (.smr).

    Part 1  read -> preprocess -> detect -> P/N split -> PCA k=2 verification
    Part 2  burst segmentation -> waveform verification

Verified on video001.SMR / Ch7 (511.06 s, 30 kHz):
    NEG 4118 (8.06 Hz) | POS-orphan 436 | 139 bursts (92.8%) | 143 clusters

Usage:
    python main.py --smr /path/video001.SMR --channel 7
    python main.py --smr ... --channel 7 --sweep       # threshold sweep + QC only

NOTE on channel identity: the SMR header carries per-channel gains in uV/bit.
Ch7 = 0.00486374, which matched the `scale` field of the Spike2 .mat export
bit-exactly and gave the highest waveform correlation (+0.85) of all 7 channels,
despite that file being *named* Ch4 and commented "from channel 1". Always
verify by gain + correlation, never by label.
"""
import argparse, os, pickle
import numpy as np

import filtering
import detection
import clustering

FS_EXPECTED   = 30000.0
NERVE_CHANNELS = (1, 2, 3, 4, 5, 6, 7)   # Ch8-10 are noise, excluded
DEFAULT_CH     = 7


# ------------------------------------------------------------------ reading
def read_smr(path, channels=NERVE_CHANNELS, cache_dir=None):
    """
    Read continuous channels from a CED SON .smr via neo (no CED DLL required).
    Returns (raw int16 (n_samples, n_ch), gains uV/bit, fs, duration_s).
    """
    from neo.rawio import Spike2RawIO
    r = Spike2RawIO(filename=path, try_signal_grouping=False)
    r.parse_header()

    sig_ch = r.header['signal_channels']
    gains = np.array([sig_ch[c - 1]['gain'] for c in channels], dtype=np.float64)
    fs = float(sig_ch[channels[0] - 1]['sampling_rate'])
    n = r.get_signal_size(0, 0, stream_index=channels[0] - 1)

    if abs(fs - FS_EXPECTED) > 1:
        print(f"  ! sampling rate {fs} Hz differs from expected {FS_EXPECTED}")

    cache = os.path.join(cache_dir, 'raw_cache.npy') if cache_dir else None
    if cache and os.path.exists(cache):
        raw = np.load(cache, mmap_mode='r')
    else:
        # NOTE: cast shape to plain ints -- numpy 2.x writes np.int64 reprs into
        # the .npy header otherwise, which np.load then refuses to parse.
        shape = (int(n), int(len(channels)))
        raw = (np.lib.format.open_memmap(cache, mode='w+', dtype=np.int16, shape=shape)
               if cache else np.empty(shape, dtype=np.int16))
        CH = 2_000_000
        for a in range(0, n, CH):
            b = min(a + CH, n)
            for j, c in enumerate(channels):
                raw[a:b, j] = r.get_analogsignal_chunk(
                    0, 0, a, b, stream_index=c - 1).ravel()
        if cache:
            raw.flush()
            del raw
            raw = np.load(cache, mmap_mode='r')
    return raw, gains, fs, n / fs


def channel_uv(raw, gains, channels, ch):
    """Extract one channel in microvolts."""
    return raw[:, list(channels).index(ch)].astype(np.float32) * gains[list(channels).index(ch)]


# ------------------------------------------------------------------ part 1
def run_part1(y_target, y_all, fs, k=detection.K_THRESH, verbose=True):
    sigma = filtering.noise_sigma(y_target)
    neg = detection.detect(y_target, fs, -1, k=k, sigma=sigma)
    pos_raw = detection.detect(y_target, fs, +1, k=k, sigma=sigma)
    pos = detection.exclude_biphasic(pos_raw, neg, fs)

    W, neg, pre, post = detection.extract_waveforms(y_all, neg, fs)
    p2p = detection.peak_to_peak(W)

    if verbose:
        dur = len(y_target) / fs
        isi = np.diff(neg) / fs
        print(f"  sigma {sigma:.3f} uV | thr {k*sigma:.2f} uV")
        print(f"  NEG {len(neg)} ({len(neg)/dur:.2f} Hz) | "
              f"POS {len(pos_raw)} -> orphan {len(pos)} ({len(pos)/dur:.2f} Hz)")
        print(f"  ISI<2ms {100*np.mean(isi<0.002):.2f}% | "
              f"ISI<1.5ms {100*np.mean(isi<0.0015):.2f}%")
    return dict(sigma=sigma, neg=neg, pos=pos, pos_raw=pos_raw,
                W=W, p2p=p2p, pre=pre, post=post)


# ------------------------------------------------------------------ part 2
def run_part2(neg_idx, W, fs, duration, verbose=True):
    times = neg_idx / fs
    order = np.argsort(times)
    bursts = clustering.segment_bursts(times[order])
    covered = sum(len(b) for b in bursts)

    if verbose:
        nb, cov = clustering.burst_null_control(times[order], duration)
        print(f"  bursts {len(bursts)} covering {covered}/{len(times)} "
              f"({100*covered/len(times):.1f}%)")
        print(f"  NULL (Poisson): {nb:.0f} bursts covering {100*cov:.1f}%")

    clusters, intruders = clustering.verify_bursts(
        bursts, times, W[:, -1, :], order=order)   # last channel = target (Ch7)
    summ = clustering.cluster_summary(clusters, intruders, len(times), covered)
    if verbose:
        print(f"  clusters {summ['n_clusters']} | spikes {summ['n_in_clusters']} | "
              f"intruders {summ['n_intruders']} | no-burst {summ['n_no_burst']}")
        print(f"  size med {summ['size_median']:.0f} max {summ['size_max']} | "
              f"within-r {summ['within_r_median']:.4f} | "
              f"ISI<2ms {summ['isi_lt_2ms_pct']:.3f}%")
    return dict(bursts=bursts, clusters=clusters, intruders=intruders,
                order=order, summary=summ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smr', required=True)
    ap.add_argument('--channel', type=int, default=DEFAULT_CH)
    ap.add_argument('--k', type=float, default=detection.K_THRESH)
    ap.add_argument('--cache', default='.')
    ap.add_argument('--out', default='sorting_result.pkl')
    ap.add_argument('--sweep', action='store_true', help='QC only: threshold sweep')
    args = ap.parse_args()

    print(f"[read] {args.smr}")
    raw, gains, fs, dur = read_smr(args.smr, NERVE_CHANNELS, args.cache)
    print(f"  {raw.shape[0]} samples, {fs:.0f} Hz, {dur:.2f} s, "
          f"channels {NERVE_CHANNELS}")
    print(f"  gains uV/bit: {np.round(gains, 6)}")

    print(f"[filter] notch 60-600 Hz Q30 -> BP {filtering.BP_LOW:.0f}-"
          f"{filtering.BP_HIGH:.0f} Hz order {filtering.BP_ORDER} -> MA{filtering.MA_TAPS}")
    filt = {}
    for c in NERVE_CHANNELS:
        filt[c] = filtering.preprocess(channel_uv(raw, gains, NERVE_CHANNELS, c), fs)
    y = filt[args.channel]
    hr = filtering.harmonic_report(y, fs)
    print("  line residual (dB re broadband): " +
          " ".join(f"{f:.0f}Hz {v:+.1f}" for f, v in hr.items()))

    if args.sweep:
        print("[QC] threshold sweep with phase-randomised surrogate")
        for row in detection.threshold_sweep(y, fs):
            print(f"  k={row['k']:.1f} thr={row['thr_uv']:6.2f} n={row['n']:6d} "
                  f"rate={row['rate']:5.2f} Hz FP~{row['fp_pct']:5.1f}% "
                  f"ISI<2ms={row['isi_lt_2ms']:5.2f}%")
        return

    print(f"[part 1] detection on Ch{args.channel}")
    y_all = np.stack([filt[c] for c in NERVE_CHANNELS if c != args.channel] +
                     [filt[args.channel]])            # target channel LAST
    p1 = run_part1(y, y_all, fs, k=args.k)

    print("[part 1b] PCA k=2 polarity verification")
    chk = detection.pn_pca_check(y, p1['neg'], p1['pos'], fs)
    print(f"  PC variance {np.round(chk['explained_variance_ratio']*100,1)}")
    print(f"  GMM agreement {chk['gmm']['agreement']*100:.1f}% | "
          f"KMeans {chk['kmeans']['agreement']*100:.1f}%")

    print("[part 2] burst clustering + waveform verification")
    p2 = run_part2(p1['neg'], p1['W'], fs, dur)

    with open(args.out, 'wb') as f:
        pickle.dump(dict(channel=args.channel, fs=fs, duration=dur,
                         gains=gains, part1=p1, part2=p2, pn_check=chk), f)
    print(f"[done] -> {args.out}")


if __name__ == '__main__':
    main()
