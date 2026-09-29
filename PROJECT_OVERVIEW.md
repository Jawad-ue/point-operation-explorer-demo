# Point Operation Explorer

## Try the live project

### [Open Point Operation Explorer](https://point-operation-explorer-demo.streamlit.app)

Point Operation Explorer is an interactive learning app for point operations in digital image processing. Upload an image, choose an operation, and apply it to see the result alongside an explanation and transformation curve.

## What you can explore

- **Image operations:** try brightness, contrast, negative, thresholding, and other point transformations.
- **Before and after:** compare the original image with the processed result.
- **Transformation curve:** see how each operation maps input brightness values to output values.
- **Image details:** inspect image statistics, histograms, and pixel values.
- **AI assistant:** ask questions about the image or operation. AI replies require an API key configured by the app owner or local user.
- **Report:** type `generate a report` in the chat for a summary of the latest result.

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. For AI assistant features, configure an API key in Streamlit secrets; image operations and local reports can be explored without an AI key.

## Project files

- `app.py` — Streamlit interface
- `operations.py` — image point operations and explanations
- `image_utils.py` — image loading, validation, and statistics
- `ai_assistant.py` — optional AI chat integration
- `report_utils.py` — report generation
