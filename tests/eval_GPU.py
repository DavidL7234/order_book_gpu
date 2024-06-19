from order_book.order_book_GPU import OrderBookTorch
import json


### just to convert an order book to JSON for evaluation
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