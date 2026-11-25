# House Prices MLOps

Modular ML pipeline predicting house prices in Tunisia (ESPRIT, 4DASI1).

## Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

## Usage
python main.py --step train
python main.py --step evaluate

## Files
- model_pipeline.py: prepare_data, train_model, evaluate_model, save_model, load_model
- main.py: CLI entry point
- data/houses.csv: Tunisian property listings (scraped from Tayara, Mubawab and others; source: BadisAbid/Tunisia_house_price_ai_model on GitHub)
