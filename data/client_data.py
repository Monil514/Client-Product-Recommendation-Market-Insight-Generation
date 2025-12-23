import pandas as pd

data = {
    "client_id": [101,102,103,104,105,106,107,108,109,110],
    "age": [35,28,50,41,30,45,26,38,34,52],
    "income": [75000,45000,120000,62000,52000,98000,40000,85000,58000,140000],
    "transactions_last_6m": [42,18,65,30,22,55,15,48,27,72],
    "avg_transaction_value": [3200,1500,5200,2800,1900,4100,1200,3600,2100,6000],
    "credit_score": [720,680,780,700,690,740,650,730,705,800],
    "product_interested": [1,0,1,0,0,1,0,1,0,1]
}

df = pd.DataFrame(data)
df.to_csv("client_data.csv", index=False)

print("CSV file created successfully")
