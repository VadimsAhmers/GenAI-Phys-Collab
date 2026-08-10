#!/usr/bin/env bash
# Overnight frequency-response campaign for the chiral mirror (needs COMSOL).
#
# Measures what the frequency-band pivot (Maksim T., 2026-07-30) assumes but
# nobody has checked: the real lineshape at his reference, whether f_res scales
# as 1/size with the waveguide fixed, whether f_res is smooth in a single
# parameter, and whether the converged eps<=10 crescent is on resonance or just
# mistuned. ~3 h at ~26 s/solve.
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

# Stage 1 (~20 min): lineshape at Maksim's reference. Gives the measured FWHM
# and Q -- so far only INFERRED as ~150 MHz from the eps sweep -- and says
# whether his reference peaks exactly at 5.8 GHz or is itself slightly detuned.
$RUN --band 5.3:6.3 --step 0.025 --out "$OUT/freq_reference.csv"

# Stage 2 (~1.5 h): all four lengths scaled together. Tests f_res ~ 1/a against
# the FIXED 32x32 mm guide -- the assumption behind "optimize anywhere in the
# band, then rescale to 5.8 GHz" -- and shows whether the peak HEIGHT changes
# with size, i.e. whether the band can reach taller peaks than 0.885.
# Band deliberately wider than 1/a predicts (5.04-6.82 GHz for these scales):
# if the fixed guide makes the shift stronger than 1/a, an extreme scale would
# otherwise peak on the band edge and that geometry's night would be wasted.
$RUN --scales 0.85,0.9,0.95,1.05,1.1,1.15 \
     --band 4.8:7.2 --step 0.1 --refine 4 --out "$OUT/freq_scale.csv"

# Stage 3 (~50 min): one parameter at a time, which is what the optimizer moves.
# f_res smooth and monotone in r and h is what makes the peak frequency usable
# as a DIRECTION in the feedback string.
$RUN --vary r_mm --values 5.4,6.6 --band 4.9:7.0 --step 0.1 --refine 4 \
     --out "$OUT/freq_axis.csv"
$RUN --vary h_mm --values 3.0,3.8 --band 4.9:7.0 --step 0.1 --refine 4 \
     --out "$OUT/freq_axis.csv"

# Stage 4 (~20 min): the converged eps<=10 crescent (|r_RR|^2 = 0.824 at 5.8).
# If its true peak sits off 5.8 GHz, then the single-frequency objective has
# been scoring a mistuned structure and the band raises the ceiling, not just
# the findability.
$RUN --base legacy --band 4.5:7.0 --step 0.05 --out "$OUT/freq_legacy.csv"

echo "=== finished $(date) ==="
