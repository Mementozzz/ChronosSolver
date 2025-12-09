# Chronos Clock Solver
Automatically solves the Chronos time-addition minigame using real-time screen capture + clock hand detection.

# NOTE BEFORE DOWNLOADING: THIS ONLY WORKS FOR 3 STAR CURRENTLY

Should work on any screen size, open an issue with your screen resolution otherwise

---

## Installation

### 1. Download a release
Go to **Releases** → download the EXE / APP / Linux binary.

OR build from source:

```bash
git clone https://github.com/Mementozzz/ChronosSolver.git
cd ChronosSolver
pip install -r requirements.txt
python main.py
```
Build as executable --> Check build_instructions.txt

---

## Usage

### Basic Usage (with GUI window)
```bash
main.exe
```
- Opens a window showing the detection status
- Press **Q** or **ESC** to exit

### Verbose Mode (with debug info)
```bash
main.exe -v
# or
main.exe --verbose
```
- Shows detailed information about each detected clock
- Useful for troubleshooting

### Headless Mode (no GUI window)
```bash
ChronosSolver.exe --no-window
```
- Runs without opening a window
- Use CTRL+C to exit

### Command Line Options
```
-v, --verbose    Enable verbose output with debug information
--no-window      Run without GUI window (headless mode)
-h, --help       Show help message
```

---

## How It Works

- Captures your screen using MSS  
- Uses HoughCircles to detect all 16 clocks automatically  
- Reads hour + minute hands  
- Computes the time sum  
- Matches it to the correct answer  
- Displays the answer in the GUI window and console

---

## Controls

- **Q** or **ESC** - Exit the program (when GUI window is active)
- **CTRL+C** - Exit the program (in console mode)
- **R** - Resume scanning for a clear frame (in case of false positives)

---

## Troubleshooting

### No clocks detected?
- Make sure the Chronos minigame is fully visible on screen
- Try running with `-v` flag to see debug information
- Check if your screen resolution is causing issues

### Program won't close?
- Press **Q** or **ESC** in the GUI window
- Or use CTRL+C in the console

---

## Disclaimer
This tool is for educational use only.  
Use at your own risk.
