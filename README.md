# AI RealTime Translator

A powerful real-time visual translation tool that captures text from your screen, detects the language, and overlays translations directly on the original content. Perfect for translating chat messages, game UI, documents, and any on-screen text.

## Features

- **Real-time Screen Capture**: Captures full screen or selected regions
- **OCR Text Extraction**: Uses Tesseract OCR to extract text with precise bounding boxes
- **Multi-language Support**: Detects and translates from any language to your target language
- **Smart Text Grouping**: Intelligently groups words into lines and chat bubbles
- **Live Overlay Display**: Shows translations overlaid on the original screenshot
- **Clipboard Monitoring**: Automatically translates text copied to clipboard
- **System Tray Integration**: Runs in background with hotkey activation
- **Configurable Settings**: Customizable fonts, colors, capture intervals, and more

## Installation

### Prerequisites

1. **Python 3.8+** installed
2. **Tesseract OCR** installed on your system
   - Windows: Download from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - Default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/AI_RealTime_Translator.git
cd AI_RealTime_Translator
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

1. Run the application:
```bash
python main.py
```

2. The application will start in your system tray
3. Use the default hotkey `Ctrl+Shift+T` to toggle translation
4. Select a screen region to capture (if not using full screen)
5. View real-time translations in the overlay window

### Configuration

The application stores settings in `~/.translator/settings.json`. Key configurable options:

- **Capture Mode**: `ocr`, `clipboard`, or `both`
- **Target Language**: Language to translate to (default: `en`)
- **Capture Interval**: Seconds between automatic captures
- **Hotkey**: Custom hotkey for toggling translation
- **Overlay Appearance**: Font size, opacity, colors, and position

### Hotkeys

- `Ctrl+Shift+T`: Toggle translation on/off
- Additional hotkeys can be configured in settings

## Project Structure

```
AI_RealTime_Translator/
├── main.py          # Main application orchestrator
├── capture.py       # Screen capture and clipboard monitoring
├── ocr.py          # OCR engine and text extraction
├── translator.py   # Language detection and translation
├── renderer.py     # Text grouping and rendering
├── overlay.py      # UI overlay and region selector
├── config.py       # Configuration management
├── requirements.txt # Python dependencies
└── README.md       # This file
```

## How It Works

1. **Capture**: Takes screenshot of screen or selected region
2. **OCR**: Extracts text with bounding boxes using Tesseract
3. **Group**: Intelligently groups words into lines and text blocks
4. **Detect**: Identifies the source language of each text block
5. **Translate**: Translates non-English text to target language
6. **Render**: Overlays translations on the original screenshot
7. **Display**: Shows translated image in real-time overlay

## Dependencies

- **Screen Capture & OCR**:
  - `mss` - Fast screen capture
  - `pytesseract` - Tesseract OCR wrapper
  - `opencv-python` - Image processing
  - `Pillow` - Image manipulation

- **Language Processing**:
  - `langdetect` - Language detection
  - `deep-translator` - Translation services

- **User Interface**:
  - `PyQt5` - GUI framework and system tray

- **Utilities**:
  - `pyperclip` - Clipboard monitoring
  - `keyboard` - Hotkey support

## Configuration File

Example `settings.json`:

```json
{
    "capture_mode": "ocr",
    "capture_interval_sec": 2.0,
    "target_language": "en",
    "overlay_font_size": 16,
    "overlay_opacity": 0.85,
    "toggle_hotkey": "ctrl+shift+t",
    "tesseract_path": "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
}
```

## Troubleshooting

### Common Issues

1. **Tesseract not found**: Ensure Tesseract OCR is installed and the path in config is correct
2. **OCR accuracy poor**: Try adjusting OCR settings or ensure text is clear and high-contrast
3. **Hotkey not working**: Check if the hotkey conflicts with other applications
4. **Performance issues**: Increase capture interval or reduce capture region size

### Logs

The application logs to console with INFO level by default. Check the console output for debugging information.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Tesseract OCR team for the excellent OCR engine
- PyQt5 developers for the GUI framework
- The open-source community for various translation APIs and libraries
