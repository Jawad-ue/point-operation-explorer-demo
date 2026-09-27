# Point Operation Explorer

A chat-style app for learning point operations in Digital Image
Processing. Upload an image, pick an operation from the compose bar at
the bottom (like picking a model), hit Apply, and the result — with
its explanation, before/after view, and the transformation curve —
appears as a message in the conversation.

## Project structure

```
point_operation_explorer/
├── app.py                          # Streamlit UI — run this
├── image_utils.py                   # loading, validation, stats, Otsu threshold
├── operations.py                    # all 6 point operations + their explanations
├── report_utils.py                  # chat-based Markdown report
├── ai_assistant.py                   # chatbot (Anthropic / OpenAI / Groq)
├── requirements.txt
├── .gitignore
├── .streamlit/
│   ├── config.toml                  # theme
│   └── secrets.toml.example         # copy → secrets.toml, fill in your key(s)
└── README.md
```

## Setup

1. Open this folder in VS Code.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up your API key** (one-time):
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   - Open `secrets.toml` and fill in the key for whichever provider you'll
     use (Anthropic, OpenAI, or Groq) — you only need one
   - `secrets.toml` is already listed in `.gitignore`, so it will **never**
     be committed or pushed to GitHub. Only the `.example` file (with no
     real key) is tracked. This means you never type your key into the
     app itself — it's loaded automatically.
5. Run the app:
   ```bash
   streamlit run app.py
   ```
   It opens at `http://localhost:8501`.

## Getting a key

- **Anthropic**: [console.anthropic.com](https://console.anthropic.com) →
  Settings → API Keys (paid — requires credits)
- **OpenAI**: [platform.openai.com](https://platform.openai.com) →
  API Keys (paid — requires credits)
- **Groq**: [console.groq.com/keys](https://console.groq.com/keys)
  (**free** tier available)

## Using the app

- Click **📎** to upload an image.
- Pick an operation from the dropdown next to it.
- If the operation has adjustable settings, a **⚙️ Params** button appears.
- Click **▶ Apply** — the result posts to the chat with a before/after
  view, a plain-English explanation, and the transformation curve
  (a graph of input brightness → output brightness, showing the actual
  shape of the math).
- Click **🔍 Detailed stats & pixel values** on any result to see the
  full stats, histograms, and a pixel value table — collapsed by
  default to keep things clean.
- Type a question in the chat box to ask the AI assistant about your
  image or the operation.
- Type something like **"generate a report"** to get a formatted
  summary of the most recent result — this is generated locally from
  the actual numbers, so it doesn't use an AI call.

## Notes

- This app is set up for **local personal use** — the sidebar shows
  whichever key is loaded from `secrets.toml` based on the provider you
  pick. If you ever want to deploy a public version for other people to
  use, that's a different setup (their usage would run on your key/
  credits), so revisit this before deploying.
- If you switch providers in the sidebar, update the **Model** field to
  match — model names differ between providers and change over time.
