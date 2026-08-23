# Hand Landmark Detection Rates on Jester Dataset: MediaPipe vs. YOLO

I just ran the calculations using the cached data we have for both models. Here is the comparison of hand landmark detection rates across the entire Jester dataset:

### 🟩 MediaPipe
* **Detected frames:** 2,566,720
* **Total frames:** 5,279,581
* **Detection Rate:** **48.62%**

### 🟦 YOLO
* **Detected frames:** 2,538,047
* **Total frames:** 5,279,544
* **Detection Rate:** **48.07%**

> [!NOTE]
> The total frames vary very slightly (~37 frames difference) likely due to a minor difference in how missing videos/frames were handled during each extraction run.

### Conclusion
Both models perform extremely similarly when it comes to simply finding any hand in the frame. **MediaPipe is marginally better by about 0.5%** (~28,000 more frames detected across the 5 million frame dataset).
