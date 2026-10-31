# v1 (2024 delivery)

The original chatbot delivered to the client: a Flask app that runs extractive question answering with BERTurk (`dbmdz/bert-base-turkish-cased`) over texts from the client's case files (`file.csv`, not included because it is confidential).

Kept unchanged for reference. See the main README for the v2 rebuild and a v1 → v2 comparison.

Run (requires the client's CSV): `pip install -r requirements.txt && python chatbot_app.py`
