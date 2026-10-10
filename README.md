# Customer Intelligence Dashboard

## Scope
This Streamlit application analyzes customer purchasing behavior in the supplied Nassau Candy Distributor transaction data. It is not a real-estate buyer or investment profiling tool.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies: `python -m pip install -r requirements.txt`
4. Start the app: `streamlit run app.py`

The `data/` directory contains the transaction data and the prepared customer segment files.

## Data-quality note
The recorded ship dates occur 904–1,642 days after order dates. The dashboard intentionally does not calculate shipping delays or use ship dates for analysis until the source is verified.

## Included features
- Dataset overview and monthly order-date sales trend
- Customer cluster counts and sales contributions
- Region and preferred-division analysis
- Model evaluation by cluster count
- Filters for cluster, main region, preferred division, and preferred ship mode
- Filtered customer data export
