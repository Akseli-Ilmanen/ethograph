# 5. Fill

Press **Fill frames between labels**. Every frame between your first and last
label gets a position; your own labels are never changed, and frames outside
that span stay empty.

| Backend | When | Cost |
|---|---|---|
| **Spline** (default) | smooth motion, labels close together | instant |
| **Optical flow** | fast turns, wingbeats — anything that is not a smooth curve between two labels | about video speed |

Neither needs a GPU. Start with the spline and switch to optical flow if the
filled points cut corners.[^fill]

[^fill]: Spline: monotone cubic interpolation {cite:p}`fritsch1980pchip`. Optical flow: pyramidal Lucas–Kanade {cite:p}`lucas1981lk,bouguet2001pyramidal`, tracked forwards and backwards across each gap.
