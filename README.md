# Tinyml_emg_bionic_hand
Design of a control system for an intelligent bionic hand prosthesis, covering EMG signal acquisition, processing, machine learning model training, and deployment on a TinyML platform.

 ---

# System Pipeline
The system follows these main steps:
1. EMG Signal Acquisition :
 * Surface electrodes capture muscle activity
2. Signal Processing :
 * band pass filtering to remove motion artifacts and offset
 * Notch filtering at 50 Hz to eliminate powerline noise
3. Dataset creation
4. ML model training :
*Linear SVM was used in this project
 * Segmentation: Sliding Window with overlap
 * Feature Extraction : time-domain features (Min,Max,Std,Mean,Mean abs)
 *Feature normalization using MinMaxScaler
5. Embedded Deployment :
 * Manual implementation of SVM decision function on ESP32
 * Real-time classification (A third rotation class is currently implemented using a predefined “101” pattern on Arduino; future work will replace this with a trained model using a second EMG channel)

---

# Related Repositories

 **Hardware Design (PCB & Schematics):**


 **EMG Database:**
 * https://github.com/SarahBounab/Read-CreateEmgData

---

# Objectives

 * Develop a low-cost bionic hand prothesis
 * Enable gesture recognition for prosthetic control ( hand opening and closing and wrist rotaion) using only two classes 0 for rest 1 for contraction 

---




