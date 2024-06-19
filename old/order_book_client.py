import time
import requests
import websocket
from typing import List, Dict

import json

import gzip
import threading
from queue import Queue

from order_book import OrderBook


class OrderBookClient:
    def __init__(self, symbols: List[str]):
        self.__base_uri = "wss://stream.binance.us:9443/ws"  # Updated WebSocket base URI
        self.__symbols: List[str] = symbols
        self.__ws = websocket.WebSocketApp(self.__base_uri,
                                           on_message=self.__on_message,
                                           on_error=self.__on_error,
                                           on_close=self.__on_close)
        self.__ws.on_open = self.__on_open
        self.__lookup_snapshot_id: Dict[str, int] = dict()
        self.__lookup_update_id: Dict[str, int] = dict()

        self.__closed = False

        self.init_snapshot = dict()
        self.order_books = dict()

        for s in self.__symbols:
            self.order_books[s] = OrderBook(s)


        self.file_handles = {}
        for s in self.__symbols:
            filename = f"{s}_order_book.bin"
            self.file_handles[s] = open(filename, 'wb')

        self.__data_queue = Queue()
        self.__storage_thread = threading.Thread(target=self.__process_data_queue)
        self.__storage_thread.start()

    def __connect(self) -> bool:
        self.__ws.run_forever()
        self.__closed = True

        self.__data_queue.put(None)
        self.__storage_thread.join()
        return True
        

    def __on_message(self, _ws, message):
        start_time = time.time()

        data = json.loads(message)
        update_id_low = data.get("U")
        update_id_upp = data.get("u")
        if update_id_low is None:
            return
        

        symbol = data.get("s")
        
        print(symbol, update_id_low, update_id_upp)
        snapshot_id = self.__lookup_snapshot_id.get(symbol)

        if snapshot_id is None:  ### Check if initial snapshot has been taken - if not: take one and listen for next message
            print("snapshot_req", symbol)
            self.init_snapshot[symbol] = self.get_snapshot(symbol)
            print("snapshot received:", symbol, self.__lookup_snapshot_id.get(symbol))
            return
        elif update_id_upp < snapshot_id + 1: ## snapshot taken way too recently - wait for new ones
            print("retry: snapshot + 1 > update_id_upp")
            return

        #self.__log_message(message)
        prev_update_id = self.__lookup_update_id.get(symbol)

        if prev_update_id is None:
            print(symbol, update_id_low, snapshot_id, update_id_upp, "(NOTE snapshot_id gets + 1)")

            assert update_id_low   <= snapshot_id + 1 <= update_id_upp
            self.__data_queue.put((self.init_snapshot[symbol], None))
            self.__data_queue.put((message, snapshot_id))
        else:
            assert update_id_low == prev_update_id + 1
            self.__data_queue.put((message, prev_update_id))

        self.__lookup_update_id[symbol] = update_id_upp

        end_time = time.time()  # End timing after message is queued
        print(f"Message handling time: {end_time - start_time} seconds;", start_time, end_time)
        return

    def __process_data_queue(self):
        while True:
            message = self.__data_queue.get()
            if message[0] is None:
                break  # Exit signal received
            start_time = time.time()
            data = json.loads(message[0])
            symbol = data.get("s")
            prev_id = message[1]
            self.__store_data(symbol, data, prev_id)

            end_time = time.time()  # End timing after message is queued
            print(f"Storage time: {end_time - start_time} seconds;", start_time, end_time)


    def __store_data(self, symbol, data, prev_id):

        if data.get("lastUpdateId"):
            self.order_books[symbol].add_snapshot(data)

        else:
            self.order_books[symbol].add_update(data, prev_id)
        self.file_handles[symbol].seek(0)
        self.file_handles[symbol].write(self.order_books[symbol].serialize())

    
    def __load_order_book(self, symbol):
        filename = f"{symbol}_order_book.bin"
        with open(filename, 'rb') as file:
            data = file.read()
            order_book = OrderBook.deserialize(data)
        return order_book

    def __on_error(self, _ws, error):
        print(f"Encountered error: {error}")
        #import traceback
        #traceback.print_exc()
        return

    def __on_close(self, _ws, _close_status_code, _close_msg):
        for file_handle in self.file_handles.values():
            file_handle.close()
        print("Connection closed")
        return

    def __on_open(self, _ws):
        print("Connection opened")
        for symbol in self.__symbols:
            _ws.send(f"{{\"method\": \"SUBSCRIBE\",  \"params\": [\"{symbol.lower()}@depth@100ms\"], \"id\": 1}}")
        return

    def __log_message(self, msg: str) -> None:
        print(msg)
        return

    def get_snapshot(self, symbol: str):
        snapshot_url = f"https://api.binance.us/api/v3/depth?symbol={symbol}&limit=1000"  # Updated API base URL
        x = requests.get(snapshot_url)
        content = x.content.decode("utf-8")
        data = json.loads(content)
        self.__lookup_snapshot_id[symbol] = data["lastUpdateId"]

        data['s'] = symbol
        return json.dumps(data)

    def start(self) -> bool:
        self.__connect()
        return True

    def stop(self) -> bool:
        self.__ws.close()
        while not self.__closed:
            time.sleep(1)
        return True


def main():
    symbols = ["BTCUSDT", "ETHUSDT"]
    orderbook_client = OrderBookClient(symbols)
    orderbook_client.start()


if __name__ == '__main__':
    main()
