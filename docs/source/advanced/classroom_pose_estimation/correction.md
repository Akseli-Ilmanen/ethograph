(target-correction-loop)=
# 6. The correction loop

The fill is a first draft. Repeat until the overlay looks right:

1. **Fill.**
2. Set the suggestion method to **Lowest fill confidence** and press
   **Suggest frames**. It brings forward the frames where the fill was least
   sure about any one keypoint.
3. Press `N` to go to the next one. **Drag** a wrong point to correct it;
   **click** a right one to accept it.
4. When a whole frame looks right, press **Approve frame** (`Shift+H`): every
   point on it becomes your label.
5. **Fill again.** Your labels and approvals stay put; only the gaps are
   recomputed.

The **conf** columns in the points table show the fill's confidence per
keypoint: 1 is a label, lower means less sure. For optical flow, the
**Disagreement tolerance** (in pixels) sets how far the forward and backward
tracks may differ before a point counts as unreliable. Raise it for large or
fast animals.
