Here is a detailed task description for the AeroEyes challenge, based on the most specific available requirements:

***

### Task Description: AeroEyes – Finding and Rescuing With AI-Powered Drones

#### Objective
Automatically locate a specified target object (e.g. backpack, person, laptop, bicycle, etc.) using aerial drone video, based on visual reference images of the target. This simulates a real-world search-and-rescue scenario, requiring robust object recognition and precise localization under realistic and challenging conditions.[1][2]

#### Provided Data
- 3 reference images of the target object (these can show the object under different viewpoints or lighting conditions).
- A drone-captured video that scans an area from an overhead vantage point, simulating the perspective and movement typical of actual aerial search missions.[1]

#### Task Requirements
1. **Object Search and Detection**
   - Your algorithm must analyze the drone footage and compare visual information against the given reference images.
   - The system should identify all locations (frames/timestamps) where the target object appears in the video, regardless of orientation, lighting, or partial occlusion.[2][1]
   - Predict the precise coordinates (bounding box or segmentation mask, as specified) for each detection.

2. **Localization and Reporting**
   - For every detected instance, report the spatial coordinates (e.g., image pixel location, possibly mapped to real-world GPS if calibration is provided).
   - Return timestamps and locations, and generate a summary output (JSON, CSV, or leaderboard format depending on competition infrastructure).

3. **Generalization and Robustness**
   - Solutions must handle variation in:
     - Object appearance (color, size, texture)
     - Camera angles and drone altitude
     - Illumination (day/night, shadows)
     - Occlusions (e.g. objects partially covered by vegetation or structures)
   - False positives should be minimized; solutions must generalize well across area scans.

4. **Evaluation Criteria**
   - **Detection accuracy:** Precision/recall and/or F1-score for finding the right object(s) per frame.
   - **Localization quality:** IoU (Intersection over Union) for bounding box/masks.
   - **Robustness:** Handling of challenging cases (e.g. partial visibility, motion blur).
   - **Efficiency:** Speed of processing, suitability for real-time or near-real-time operation (depending on the challenge phase).[2]

#### Constraints and Rules
- Only provided visual references are permissible; participants may not manually annotate target objects in the video.
- No external human intervention during inference.
- Use of external datasets and pretrained models may be subject to the competition’s open/closed track rules.

#### Sample Workflow
1. **Preprocessing:** Normalize video frames, enhance image quality, synchronize reference image features.
2. **Feature Extraction:** Use deep learning models (such as vision transformers, CNN-based object detectors) to encode both reference images and video frames.
3. **Matching & Localization:**
   - Apply cross-attention or similarity matching between frame features and reference object tokens.
   - Predict bounding boxes or segmentation masks for detected instances.
   - Aggregate results across video frames and deduplicate repeated sightings.
4. **Submission:** Output detection results per frame in the prescribed format.

#### Technologies and Methods
- Convolutional Neural Networks (CNNs), Vision Transformers
- Object Re-identification, Cross-Attention Mechanisms
- Real-time inference optimizations
- Data augmentation to simulate varied environments

#### Real-World Relevance
- The challenge reflects critical real-world needs in search-and-rescue, disaster response, and rapid area scanning where drones and AI must collaborate to locate missing persons or essential items quickly and reliably.[2]

#### Success Criteria
- High retrieval rates with minimal false alarms
- Consistent localization of target objects across diverse real-world scenes
- Scalable, deployable solution for actual emergency or field use cases

***

This detailed description covers the input/output, expected algorithm performance, evaluation, and scenario relevance for AeroEyes. Let me know if you need example annotation formats, baseline model recommendations, or further clarification.

[1](https://challenge.zalo.ai/portal/aero-eyes)
[2](https://www.vietnam.vn/en/khoi-dong-dau-truong-tri-tue-nhan-tao-zalo-ai-challenge-2025)
[3](https://www.facebook.com/groups/township.official/posts/2412333525692092/)
[4](https://www.science.edu.sg/docs/default-source/default-document-library/drone-odyssey-challenge-2023-v1-167f18299a7f64ad094accc008ee856ea.pdf)
[5](https://ntu-aris.github.io/caric/)
[6](https://droniada.eu/wp-content/uploads/2023/02/Rules-of-the-Droniada-Challenge-2023-eng.pdf)
[7](https://www.aposteriori.com.sg/wp-content/uploads/2018/05/DroneOdyssey_Manual.pdf)
[8](https://arxiv.org/html/2406.01029v1)
[9](https://recf.org/documents/2022/09/aerial-drone-competition-mission-2023-blackout-game-manual.pdf)
[10](https://www.scribd.com/document/925760338/Challenge-Booklet)