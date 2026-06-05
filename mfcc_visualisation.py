""" 
MFCC Visualization - FIXED VERSION 
==================================== 
Generates: 
  1. mfcc_scatter_plot.png   - PCA scatter of features 
  2. frs_comparison.png      - Before vs After FRS per classs 
  3. mfcc_heatmap.png        - Mean MFCC per class heatmap 
 
Run from the same folder as your sounds/ directory. 
""" 
 
import numpy as np 
import matplotlib 
matplotlib.use('Agg')  # non-interactive backend - works everywhere 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec 
import librosa 
import librosa.display 
import os 
import glob 
from sklearn.decomposition import PCA 
 
# ── CONFIG 
─────────────────────────────────────────────────────────
─────────── 
COMMANDS      = ['faaah', 'click', 'whistle', 'pop', 'hiss', 'hum'] 
SAMPLE_RATE   = 22050 
N_MFCC        = 13 
FRS_THRESHOLD = 0.6 
COLORS        = ['#1D9E75', '#534AB7', '#D85A30', '#BA7517', '#E24B4A', '#0F6E56'] 
 
# ── AUTO-DETECT DATA PATH 
───────────────────────────────────────────────────── 
def find_data_path(): 
    candidates = [ 
        "sounds/train", 
        "sounds/clean_train", 
        "../sounds/train", 
        "Non-Linguistic-Vocal-Command-Recognition-System-main/sounds/train", 
    ] 
    for path in candidates: 
        # Check if at least one command folder exists inside 
        for cmd in COMMANDS: 
            if os.path.isdir(os.path.join(path, cmd)): 
                print(f"  Found data at: {os.path.abspath(path)}") 
                return path 
    return None 
 
# ── FEATURE EXTRACTION 
──────────────────────────────────────────────────────── 
def extract_features_full(audio_file): 
    """Returns feature vector, frs_score, raw mfcc, energy array""" 
    try: 
        y, sr = librosa.load(audio_file, sr=SAMPLE_RATE, duration=2.0) 
        y, _  = librosa.effects.trim(y, top_db=25) 
        if np.max(np.abs(y)) > 0: 
            y = y / np.max(np.abs(y)) 
 
        mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, 
                                       n_fft=2048, hop_length=512) 
        energy = np.sum(mfcc**2, axis=0) 
        if np.max(energy) > 0: 
            energy = energy / np.max(energy) 
 
        mask = energy > FRS_THRESHOLD 
        mfcc_frs = mfcc[:, mask] if mask.sum() >= 5 else mfcc 
 
        frs_score = (np.mean(energy[mask]) if mask.sum() > 0 
                     else np.mean(energy)) 
 
        return np.mean(mfcc_frs, axis=1), frs_score, mfcc, energy 
    except Exception as e: 
        print(f"    [ERROR] {audio_file}: {e}") 
        return None 
 
# ── LOAD DATASET 
─────────────────────────────────────────────────────────
───── 
def load_dataset(data_path): 
    features, labels = [], [] 
    per_class_files  = {}   # {label: [filepath, ...]} 
 
    for lbl, cmd in enumerate(COMMANDS): 
        folder = os.path.join(data_path, cmd) 
        wavs   = glob.glob(os.path.join(folder, "*.wav")) 
 
        if not wavs: 
            print(f"  [WARN] No .wav files in: {folder}") 
            per_class_files[lbl] = [] 
            continue 
 
        per_class_files[lbl] = wavs 
        print(f"  {cmd}: {len(wavs)} files") 
 
        for fp in wavs: 
            result = extract_features_full(fp) 
            if result is not None: 
                fv, _, _, _ = result 
                features.append(fv) 
                labels.append(lbl) 
 
    return np.array(features), np.array(labels), per_class_files 
 
# 
═════════════════════════════════════════════════════════
════════════════════ 
# FIGURE 1 — PCA Scatter 
# 
═════════════════════════════════════════════════════════
════════════════════ 
def plot_scatter(features, labels): 
    print("\n[1/3] Generating PCA scatter plot...") 
    pca     = PCA(n_components=2) 
    reduced = pca.fit_transform(features) 
    var     = pca.explained_variance_ratio_ 
 
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor='white') 
    ax.set_facecolor('#F9F9F9') 
 
    for lbl, (cmd, col) in enumerate(zip(COMMANDS, COLORS)): 
        mask = labels == lbl 
        if mask.sum() == 0: 
            continue 
        ax.scatter(reduced[mask, 0], reduced[mask, 1], 
                   c=col, label=cmd, s=72, alpha=0.85, 
                   edgecolors='white', linewidths=0.6) 
        cx = reduced[mask, 0].mean() 
        cy = reduced[mask, 1].mean() 
        ax.annotate(cmd, (cx, cy), fontsize=8.5, fontweight='bold', 
                    color=col, ha='center', va='bottom', 
                    xytext=(0, 9), textcoords='offset points') 
 
    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}% variance)', fontsize=10) 
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}% variance)', fontsize=10) 
    ax.set_title('PCA projection of MFCC-FRS feature vectors\n' 
                 '(6 non-linguistic sound classes)', fontsize=11, pad=10) 
    ax.legend(title='Sound class', fontsize=8, title_fontsize=9, 
              loc='best', framealpha=0.9, edgecolor='#cccccc') 
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5) 
    ax.spines[['top', 'right']].set_visible(False) 
 
    plt.tight_layout() 
    plt.savefig('mfcc_scatter_plot.png', dpi=300, bbox_inches='tight', 
                facecolor='white') 
    plt.close() 
    print("  Saved: mfcc_scatter_plot.png") 
 
# 
═════════════════════════════════════════════════════════
════════════════════ 
# FIGURE 2 — FRS Before vs After 
# 
═════════════════════════════════════════════════════════
════════════════════ 
def plot_frs_comparison(per_class_files): 
    print("\n[2/3] Generating FRS before/after comparison...") 
 
    fig = plt.figure(figsize=(15, 5), facecolor='white') 
    fig.suptitle( 
        'MFCC frames: all frames (top) vs FRS-filtered frames (bottom)  [θ = 0.6]', 
        fontsize=11, y=1.02 
    ) 
 
    gs = gridspec.GridSpec(2, len(COMMANDS), 
                           figure=fig, hspace=0.55, wspace=0.35) 
 
    any_plotted = False 
 
    for col_idx, (cmd, col) in enumerate(zip(COMMANDS, COLORS)): 
        lbl   = col_idx 
        wavs  = per_class_files.get(lbl, []) 
 
        if not wavs: 
            # empty axes with a note 
            for row in range(2): 
                ax = fig.add_subplot(gs[row, col_idx]) 
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                        transform=ax.transAxes, fontsize=8, color='gray') 
                ax.set_xticks([]); ax.set_yticks([]) 
                if row == 0: 
                    ax.set_title(cmd, fontsize=9, fontweight='bold', color=col) 
            continue 
 
        fp = wavs[0]   # use first file of each class 
        result = extract_features_full(fp) 
        if result is None: 
            continue 
 
        _, _, mfcc, energy = result 
        mask     = energy > FRS_THRESHOLD 
        mfcc_frs = mfcc[:, mask] if mask.sum() >= 5 else mfcc 
        kept_pct = mask.sum() / len(mask) * 100 
 
        vmin = mfcc.min() 
        vmax = mfcc.max() 
 
        # ── Row 0: all frames 
────────────────────────────────────────── 
        ax0 = fig.add_subplot(gs[0, col_idx]) 
        img0 = ax0.imshow(mfcc, aspect='auto', origin='lower', 
                          cmap='coolwarm', vmin=vmin, vmax=vmax, 
                          interpolation='nearest') 
        ax0.set_title(cmd, fontsize=9, fontweight='bold', color=col) 
        ax0.set_xlabel('Frames', fontsize=7) 
        if col_idx == 0: 
            ax0.set_ylabel('MFCC coeff.', fontsize=7) 
            ax0.text(-0.38, 0.5, 'All frames', 
                     transform=ax0.transAxes, fontsize=8, va='center', 
                     ha='center', rotation=90, color='#444') 
        else: 
            ax0.set_yticks([]) 
        ax0.tick_params(labelsize=6) 
 
        # ── Row 1: FRS-filtered 
──────────────────────────────────────── 
        ax1 = fig.add_subplot(gs[1, col_idx]) 
        ax1.imshow(mfcc_frs, aspect='auto', origin='lower', 
                   cmap='coolwarm', vmin=vmin, vmax=vmax, 
                   interpolation='nearest') 
        ax1.set_xlabel(f'{kept_pct:.0f}% kept', fontsize=7) 
        if col_idx == 0: 
            ax1.set_ylabel('MFCC coeff.', fontsize=7) 
            ax1.text(-0.38, 0.5, 'FRS filtered', 
                     transform=ax1.transAxes, fontsize=8, va='center', 
                     ha='center', rotation=90, color='#444') 
        else: 
            ax1.set_yticks([]) 
        ax1.tick_params(labelsize=6) 
 
        any_plotted = True 
 
    if not any_plotted: 
        print("  [WARN] No audio files found for FRS comparison.") 
        plt.close() 
        return 
 
    plt.savefig('frs_comparison.png', dpi=300, bbox_inches='tight', 
                facecolor='white') 
    plt.close() 
    print("  Saved: frs_comparison.png") 
 
# 
═════════════════════════════════════════════════════════
════════════════════ 
# FIGURE 3 — Mean MFCC Heatmap 
# 
═════════════════════════════════════════════════════════
════════════════════ 
def plot_mfcc_heatmap(features, labels): 
    print("\n[3/3] Generating mean MFCC heatmap...") 
 
    mean_mfcc = np.zeros((N_MFCC, len(COMMANDS))) 
    for lbl in range(len(COMMANDS)): 
        mask = labels == lbl 
        if mask.sum() > 0: 
            mean_mfcc[:, lbl] = features[mask].mean(axis=0) 
 
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor='white') 
 
    im = ax.imshow(mean_mfcc, aspect='auto', cmap='RdBu_r', 
                   interpolation='nearest') 
 
    ax.set_xticks(range(len(COMMANDS))) 
    ax.set_xticklabels(COMMANDS, fontsize=10) 
    ax.set_yticks(range(N_MFCC)) 
    ax.set_yticklabels([f'MFCC {i+1}' for i in range(N_MFCC)], fontsize=8) 
    ax.set_xlabel('Sound class', fontsize=11) 
    ax.set_ylabel('MFCC coefficient index', fontsize=11) 
    ax.set_title('Mean FRS-filtered MFCC coefficients per sound class', 
                 fontsize=11, pad=12) 
 
    # Annotate each cell 
    std = mean_mfcc.std() 
    for i in range(N_MFCC): 
        for j in range(len(COMMANDS)): 
            val     = mean_mfcc[i, j] 
            txt_col = 'white' if abs(val) > std * 1.1 else '#222222' 
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                    fontsize=6.5, color=txt_col) 
 
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02) 
    cbar.set_label('Mean coefficient value', fontsize=9) 
    cbar.ax.tick_params(labelsize=8) 
 
    plt.tight_layout() 
    plt.savefig('mfcc_heatmap.png', dpi=300, bbox_inches='tight', 
                facecolor='white') 
    plt.close() 
    print("  Saved: mfcc_heatmap.png") 
 
# 
═════════════════════════════════════════════════════════
════════════════════ 
# MAIN 
# 
═════════════════════════════════════════════════════════
════════════════════ 
if __name__ == "__main__": 
    print("=" * 55) 
    print("  MFCC VISUALIZATION  —  FIXED VERSION") 
    print("=" * 55) 
 
    # 1. Find data 
    data_path = find_data_path() 
    if data_path is None: 
        print("\n[ERROR] Could not find sounds/train folder.") 
        print("Make sure you run this script from your project root,") 
        print("and that sounds/train/<command>/*.wav files exist.") 
        exit(1) 
 
    # 2. Load 
    print(f"\nLoading dataset from: {data_path}") 
    features, labels, per_class_files = load_dataset(data_path) 
 
    if len(features) == 0: 
        print("\n[ERROR] No features extracted. Check your .wav files.") 
        exit(1) 
 
    print(f"\nTotal samples loaded: {len(features)}") 
    for lbl, cmd in enumerate(COMMANDS): 
        count = (labels == lbl).sum() 
        print(f"  {cmd}: {count} samples") 
 
    # 3. Plot all three figures 
    plot_scatter(features, labels) 
    plot_frs_comparison(per_class_files) 
    plot_mfcc_heatmap(features, labels) 
 
    print("\n" + "=" * 55) 
    print("  ALL 3 FIGURES SAVED SUCCESSFULLY") 
    print("=" * 55) 
    print("\n  mfcc_scatter_plot.png  -> Section IV.C") 
    print("  frs_comparison.png     -> Section III.D") 
    print("  mfcc_heatmap.png       -> Section IV.C") 
    print("\nInsert these into your Word document as figures.")