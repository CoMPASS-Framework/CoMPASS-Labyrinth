# EEG Preprocessing

## EDF-to-Numpy Converter

A modular toolkit for converting EEG/physiology recordings stored in .edf format into NumPy .npy dictionaries for efficient downstream analysis in Python. This converter captures raw signals, signal metadata, and file headers from multi-channel EDF recordings and saves them as structured .npy objects.

---

## Core Capabilities

### - Single File Conversion
- Convert individual .edf files to .npy format using pyedflib.

### - Batch Processing
- Recursively convert multiple .edf files in a directory, with pattern matching and overwrite options.

### - Signal Metadata Extraction
- Extract headers, sample rates, signal labels, and physical units per channel.

### - Easy Loading
- Load back .npy files into dictionaries using load_converted_edf().
