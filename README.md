# DAW-GESTURE-CONTROL

## SUMMARY
DAW-Gesture-Control is a vision-based system that allows musicians to control a DAW using hand gestures instead of physical MIDI controllers

## DATASET
Collected images of hand gestures specifically - two,four,closed,open hand,closed hand,nothing dataset.
Two - hand displaying two
Four - hand displaying four
Closed - closed hand
Open - open hand
Noting - dataset containing gestures that are flagged as nothing by the CNN

## HARDWARE REQUIRED
A laptop, an instrument, an interface, a working camera.

## PIPELINE
# CAMERA
A camera source is required to capture the frames.

# MEDIAPIPE
Mediapipe extracts a hand crop of the image along with the landmarks.

# INTENT-CONTROL CNN
The cnn decides whether the gesture being performed by the musician is intentional or not. If it's intentional, then send the hand landmarks to a decision tree for gesture recognition.

# DECISION TREE
The decision tree decides what type of gesture it is.

# GEOMETRICAL DECISIONS
For the gestures of one and three, I am using geometry(hand landmarks from mediapipe) of the hand and a series of if-else conditions concerning the pip joint of the hand to identify the gesture and provide a secondary confidence measure of existing gesture.

# MIDI OUTPUT
The program generates midi output which is sent to a fake MIDI port generated with the help of loopMIDI. The loopMIDI port then sends the output to the DAW just like a MIDI controller would.

## ADVANTAGES
# IMPROVED WORFLOW
With gestures, recording at home, switches patches,etc become a lot quicker.
# FLEXIBILITY
The gestures can be mapped to whatever you want as it essentially functions like a MIDI controller with gestures instead.
# NO HARDWARE
This project, if refined could potentially replace a pedal for live performances and all it needs is a camera, laptop and interface.

## LIMITATIONS AND FURTHER SCOPE
# FPS
Fps struggles as there's 4 levels of processing.
# LIMITED GESTURES
Currently there's only 5 gestures, as the open hand gesture is used for activating the gesture control system. For more gestures, a larger dataset is required. Furthermore, instead of just static gestures, dynamic gestures can also be added for increasing volume, panning, faders,etc.

## APPLICATIONS
The project can be used in a studio or live environment for musicians to record, switch patches, and work with a daw with using hand gestures. Rather than continuosly pressing the record button while getting the perfect take, using hand gestures improve workflow. 


