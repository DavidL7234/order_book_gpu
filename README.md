# Order Book GPU

Order Book GPU is a high-performance Python library designed for managing and analyzing order books in financial markets, leveraging GPU acceleration to handle high-frequency data efficiently.

## Features

- Real-time order book updates
- GPU-accelerated computations for rapid processing
- Customizable statistics and metrics computation
- Support for multiple financial instruments

## Features
Evaluation: refer to order_book_gpu/tests/

Extract data from order_book_gpu/tests/compressed.zip

### Markowitz
eval_markowitz.py

The following packages are required:
````python
python -m pip install scipy
python -m pip install PyPortfolioOpt
````

### Single-Period Optimization (SPO) and Multi-Period Optimization (MPO)
eval_cvx.py
eval_multi_cvx.py

The following packages is required:
````python
python -m pip install cvxportfolio
````

## Installation
Order Book GPU can be installed using pip.

````python
git clone https://github.com/yourusername/order-book-gpu.git
cd order-book-gpu
pip install .
````

## To Run

````python
from order_book import OrderBookClientGPU
symbols = ["BTCUSDT", "ETHUSDT"]
orderbook_client = OrderBookClientGPU(symbols, custom_stat_fn=None, path='.')
orderbook_client.start()
````

## Opening and Reading Order Books: Refer to tests/eval_GPU.py

````python
from order_book import OrderBookTorch
import json


# just to convert an order book to JSON for evaluation
s = "ETHUSDT"
filename = f"{s}_order_book.bin"

with open(filename, 'rb') as file:
    data = file.read()
    order_book = OrderBookTorch.deserialize(data)

saved_data = {update_id: {
            'bids': frame['bids'].cpu().tolist(),
            'asks': frame['asks'].cpu().tolist(),
            'stats': frame['stats']
        } for update_id, frame in order_book.frames.items()}

with open(filename.split('.')[0]+'.json', 'ab') as file2:
    file2.write(json.dumps(saved_data).encode('utf-8'))

print(order_book.frames.keys())
````
