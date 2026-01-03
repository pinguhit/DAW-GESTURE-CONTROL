import mido

midi_out = None


def init_midi(MIDI_PORT_NAME):
    global midi_out

    ports = mido.get_output_names()

    for name in ports:
        if MIDI_PORT_NAME == name:
            midi_out = mido.open_output(name)
            print(f"MIDI connected: {name}")
            return

    raise RuntimeError(
        f"MIDI port '{MIDI_PORT_NAME}' not found.\n"
        f"Available ports: {ports}"
    )


def send_note(note, velocity=100, channel=0):
    if midi_out is None:
        raise RuntimeError("MIDI not initialized. Call init_midi() first.")

    midi_out.send(
        mido.Message(
            "note_on",
            note=note,
            velocity=velocity,
            channel=channel
        )
    )

def send_cc(cc, value, channel=0):
    if midi_out is None:
        raise RuntimeError("MIDI not initialized")

    midi_out.send(
        mido.Message(
            "control_change",
            control=cc,
            value=value,
            channel=channel
        )
    )


def close_midi():
    global midi_out

    if midi_out is not None:
        try:
            midi_out.close()
        except Exception as e:
            print("Error closing MIDI:", e)
        finally:
            midi_out = None