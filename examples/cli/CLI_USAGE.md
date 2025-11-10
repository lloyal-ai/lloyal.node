# CLI Usage

## Basic Usage

Run the chat CLI:

```bash
npm start
```

This will show a **Boot Screen** where you can configure:
- **Model**: Path or URL to the model file
- **Context Size**: Default 8192
- **Threads**: Default 1

Use **Tab**, **Up Arrow**, or **Down Arrow** to navigate between fields. Press **Enter** to start loading the model.

## Boot Screen

The boot screen displays on startup and allows you to review/edit configuration before loading:

```
┌─ Configuration ─┐
│                 │
└─────────────────┘

Model (path or URL):
  [current model path or URL]

Context Size: (default 8192)
  [8192]

Threads: (default 1)
  [1]

     [Start]

Tab/↑/↓ to navigate • Enter to start
```

**Features:**
- Edit any field before starting
- Values from command-line arguments pre-populate the fields
- Green highlighting shows which field is focused
- Press Tab, Down Arrow (↓), or Up Arrow (↑) to navigate between fields
- Press Cmd/Ctrl+Backspace to clear the focused field
- Press Enter from any field to start loading

## Command Line Arguments

### --model

Specify a model path (local file or HTTP/HTTPS URL):

```bash
# Local model path
npm start -- --model=/path/to/model.gguf

# Remote model URL (downloads to ~/.clai/models)
npm start -- --model=https://huggingface.co/lmstudio-community/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf
```

### --context

Specify context window size (default: 2048):

```bash
npm start -- --model=/path/to/model.gguf --context=4096
```

### --threads

Specify number of threads (default: 4):

```bash
npm start -- --model=/path/to/model.gguf --threads=8
```

## Combined Example

```bash
npm start -- \
  --model=https://huggingface.co/lmstudio-community/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  --context=2048 \
  --threads=4
```

## Download Behavior

- Models from URLs are downloaded to `~/.clai/models/`
- If a model is already cached, it won't be re-downloaded
- Download progress is shown with a visual progress bar
- Download speed and size information is displayed in real-time

## Model Loading

After download (or when using local models), the loading screen shows:
- Model path
- Loading progress with spinner
- llama.cpp initialization logs (in small, dimmed text)
- GPU information and configuration details

## Keyboard Shortcuts

During chat:
- **Ctrl+C**: Exit the application
- **Space**: Pause/resume generation
- **Up/Down arrows**: Navigate through prompt history
- **Ctrl/Cmd+Backspace**: Clear current input line
- **Tab**: Switch focus between input and Clear button
- **Enter** (on Clear button): Clear chat history

## Directory Structure

```
~/.clai/
└── models/              # Downloaded models cache
    ├── model1.gguf
    └── model2.gguf
```
