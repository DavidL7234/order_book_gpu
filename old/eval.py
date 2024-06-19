from order_book import OrderBook
import json

s = "ETHUSDT"
filename = f"{s}_order_book.bin"

with open(filename, 'rb') as file:
    data = file.read()
    order_book = OrderBook.deserialize(data)

with open(filename.split('.')[0]+'.json', 'ab') as file2:
    file2.write(json.dumps(order_book.frames).encode('utf-8'))

print(order_book.frames.keys())