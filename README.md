# Chronos Clock Solver
Automatically solves the Chronos time-addition minigame using real-time screen capture + clock hand detection.

# NOTE BEFORE DOWNLOADING: THIS ONLY WORKS FOR 3 STAR CURRENTLY

Should work on any screen size, open an issue with your screen resolution otherwise

---

## Installation

### 1. Download a release
Go to **Releases** → download the EXE / APP / Linux binary.

OR build from source:

```
git clone https://github.com/Mementozzz/ChronosSolver.git
cd ChronosSolver
pip install -r requirements.txt
python main.py
```

---

## How It Works

- Captures your screen using MSS  
- Uses HoughCircles to detect all 16 clocks automatically  
- Reads hour + minute hands  
- Computes the time sum  
- Matches it to the correct answer  
- Prints the correct answer index  

---

## Disclaimer
This tool is for educational use only.  
Use at your own risk.
