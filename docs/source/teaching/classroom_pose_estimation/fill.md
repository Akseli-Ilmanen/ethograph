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

[^fill]: Spline: monotone cubic interpolation ([`scipy.interpolate.PchipInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)). Optical flow: pyramidal Lucas–Kanade ([`cv2.calcOpticalFlowPyrLK`](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323)), tracked forwards and backwards across each gap.
