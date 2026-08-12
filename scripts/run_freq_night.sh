#!/usr/bin/env bash
# Overnight frequency-response campaign for the chiral mirror (needs COMSOL).
#
# Measures what the frequency-band pivot (Maksim T., 2026-07-30) assumes but
# nobody has checked: the real lineshape at his reference, whether f_res scales
# as 1/size with the waveguide fixed, whether f_res is smooth in a single
# parameter, and whether the converged eps<=10 crescent is on resonance or just
# mistuned. ~3.5 h at ~26 s/solve.
#
# GRID: the three points from --check (0.0388 / 0.8851 / 0.0011 at 5.6/5.8/6.0)
# imply a line well under the 151 MHz we had INFERRED from the eps sweep -- one
# side reads ~86 MHz, the other ~14 MHz. So everything here runs on a 25 MHz grid
# (5 MHz for the lineshape), and the size scan uses a narrow window around the
# PREDICTED resonance instead of a coarse scan of the whole band: a 100 MHz grid
# would step straight over a line this narrow and report a tail sample as "the
# peak".
#
# BAND: the 32x32 mm guide is single-mode only between 4.684 and 6.625 GHz. Above
# that the TE11/TM11 pair also propagates and a 2-mode-per-port S-matrix stops
# accounting for the power; just above the lower cutoff |r| goes to 1 for any
# structure at all. sweep_freq.py drops points outside ~4.87-6.56 GHz, which also
# caps the scale range below to ~+-12%.
#
# Every stage appends to its own CSV and skips points already there, so this
# script is safe to re-run after an interruption -- it resumes where it stopped.
# Stages are ordered by value, so a night that gets cut short still delivers.
#
#   bash scripts/run_freq_night.sh [OUTDIR]        # default OUTDIR: ~/Downloads/freq
set -u

OUT="${1:-$HOME/Downloads/freq}"
mkdir -p "$OUT"
RUN="uv run python scripts/sweep_freq.py"
LOG="$OUT/night.log"

echo "writing to $OUT (log: $LOG)"
exec > >(tee -a "$LOG") 2>&1
echo "=== started $(date) ==="

# Stage 0: does the frequency actually reach the solve? Nothing below is
# meaningful if this fails, so stop rather than burn the night on flat lines.
if ! $RUN --check; then
  echo "ABORT: frequency plumbing check failed (see above)."
  exit 1
fi

# Stage 1 (~20 min): the line at Maksim's reference, 25 MHz over 1 GHz. Shows the
# shape -- symmetric Lorentzian, or the asymmetry the three check points hint at.
$RUN --band 5.3:6.3 --step 0.025 --out "$OUT/freq_reference.csv"

# Stage 1b (~26 min): the same line at 5 MHz. This is the number to quote for Q
# and, through stage 2's df/da, for the fabrication tolerances -- the 151 MHz we
# sent Maksim came from an eps grid whose step was itself ~145 MHz, so it was an
# upper bound at the resolution limit rather than a measurement.
$RUN --band 5.65:5.95 --step 0.005 --out "$OUT/freq_reference.csv"

# Stage 2 (~55 min): all four lengths scaled together, each on its own +-300 MHz
# window around the predicted 5.8/a. Tests f_res ~ 1/a against the FIXED guide --
# the assumption behind "optimize anywhere in the band, then rescale to 5.8" --
# and whether the peak HEIGHT moves with size. --no-anchor: the unscaled point is
# already stage 1. Scales stay within +-12% so the predicted peak stays inside
# the single-mode window.
$RUN --scales 0.93,0.96,1.04,1.08,1.12 --no-anchor \
     --window 0.3 --step 0.025 --out "$OUT/freq_scale.csv"

# Stage 3 (~78 min): one parameter at a time, which is what the optimizer moves.
# f_res smooth and monotone in r and h is what makes the peak frequency usable as
# a DIRECTION in the feedback string. Single-axis moves shift f_res less than a
# full rescale, so one shared band is enough here.
$RUN --vary r_mm --values 5.4,6.6 --band 5.3:6.4 --step 0.025 \
     --out "$OUT/freq_axis.csv"
$RUN --vary h_mm --values 3.0,3.8 --band 5.3:6.4 --step 0.025 \
     --out "$OUT/freq_axis.csv"

# Stage 4 (~28 min): the converged eps<=10 crescent (|r_RR|^2 = 0.8175 at 5.8;
# the 0.824 this line used to quote was a stray number, see LEGACY_BEST).
# If its true peak sits off 5.8 GHz, the single-frequency objective has been
# scoring a mistuned structure and the band raises the ceiling, not just the
# findability. Its geometric basin was ~6x wider relatively, so 25 MHz is ample.
$RUN --base legacy --band 4.9:6.5 --step 0.025 --out "$OUT/freq_legacy.csv"

echo "=== finished $(date) ==="
