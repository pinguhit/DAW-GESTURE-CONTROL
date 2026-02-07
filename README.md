# 🎛️ DAW-GESTURE-CONTROL

## 📌 SUMMARY
DAW-Gesture-Control is a vision-based system that allows musicians to control a DAW using hand gestures instead of physical MIDI controllers.
Demo Video -> https://youtu.be/wbT2AZZo_jE

---

## 📂 DATASET
Collected images of hand gestures specifically:
- **Two** – hand displaying two  
- **Four** – hand displaying four  
- **Closed** – closed hand  
- **Open** – open hand  
- **Nothing** – dataset containing gestures that are flagged as nothing by the CNN  

---

## 🧰 HARDWARE REQUIRED
- A laptop  
- An instrument  
- An interface  
- A working camera  

---

## 🔁 PIPELINE

### CAMERA
A camera source is required to capture the frames.

### MEDIAPIPE
MediaPipe extracts a hand crop of the image along with the landmarks.

### INTENT-CONTROL CNN
The CNN decides whether the gesture being performed by the musician is intentional or not.  
If it's intentional, the hand landmarks are sent to a decision tree for gesture recognition.

### DECISION TREE
The decision tree decides what type of gesture it is.

### GEOMETRICAL DECISIONS
For the gestures of one and three, geometry (hand landmarks from MediaPipe) and a series of if-else conditions concerning the PIP joint of the hand are used to identify the gesture and provide a secondary confidence measure of the existing gesture.

### MIDI OUTPUT
The program generates MIDI output which is sent to a fake MIDI port generated with the help of **loopMIDI**.  
The loopMIDI port then sends the output to the DAW just like a MIDI controller would.

---

## ✅ ADVANTAGES

### IMPROVED WORKFLOW
With gestures, recording at home and switching patches become a lot quicker.

### FLEXIBILITY
The gestures can be mapped to whatever you want, as it essentially functions like a MIDI controller using gestures.

### NO HARDWARE
This project, if refined, could potentially replace a pedal for live performances and only requires a camera, laptop, and interface.

---

## ⚠️ LIMITATIONS AND FURTHER SCOPE

### FPS
FPS struggles as there are four levels of processing.

### LIMITED GESTURES
Currently there are only five gestures, as the open hand gesture is used for activating the gesture control system.  
For more gestures, a larger dataset is required. Furthermore, instead of just static gestures, dynamic gestures can also be added for increasing volume, panning, faders, etc.

---

## 🎵 APPLICATIONS
The project can be used in a studio or live environment for musicians to record, switch patches, and work with a DAW using hand gestures.  
Rather than continuously pressing the record button while getting the perfect take, using hand gestures improves workflow.
