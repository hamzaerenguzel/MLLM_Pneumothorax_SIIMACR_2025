# MLLM Pneumothorax Detection on SIIM-ACR Dataset
This repository contains the Python scripts used in the study evaluating multimodal large language models (GPT-4o, Gemini 2, and Claude 4) for pneumothorax detection on the SIIM-ACR dataset.

## Files
- `dicom_to_png.py` — DICOM to PNG preprocessing pipeline  
- `gpt_xray.py` — GPT-4o API call logic    
- `gemini_xray.py` — Gemini 2 API call logic  
- `claude_xray.py` — Claude 4 API call logic  
- `requirements.txt` — Python dependencies  

## Requirements
Python ≥ 3.9  
Install dependencies:
```bash
pip install -r requirements.txt
