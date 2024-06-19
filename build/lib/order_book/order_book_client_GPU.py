import time
import requests
import websocket
from typing import List, Dict

import json

import gzip

from multiprocessing import Process, Queue as MPQueue

from .order_book_GPU import OrderBookTorch

import torch

import os

class OrderBookClientGPU:
    """
    A client for managing order books of multiple trading symbols using WebSocket to receive real-time updates
    and processes to handle the data efficiently.

    Attributes:
        symbols (List[str]): List of trading symbols to manage order books for.
        custom_stat_fn (callable): A custom function for additional statistics calculation.
    """

    def __init__(self, symbols: List[str], custom_stat_fn=None, path='.'):
        self.__base_uri = "wss://stream.binance.us:9443/ws"
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

        self.custom_stat_fn = custom_stat_fn
        self.data_queues = {symbol: MPQueue() for symbol in symbols}
        self.processes = {}

        for symbol in symbols:
            p = Process(target=self.__process_symbol_data, args=(symbol, self.data_queues[symbol], path))
            p.start()
            self.processes[symbol] = p

    def __connect(self) -> bool:
        """
        Establishes the WebSocket connection and initializes the data processing.

        Returns:
            bool: True if connection process is completed successfully.
        """
        self.__ws.run_forever()
        self.__closed = True

        for q in self.data_queues.values():
            q.put(None)
        for p in self.processes.values():
            p.join()

        return True
        

    def __on_message(self, _ws, message):
        """
        Handles incoming WebSocket messages, parsing and dispatching them to appropriate queues for processing.
        Retrieves an initial snapshot before adding updates to order book.

        Args:
            _ws (WebSocketApp): The WebSocket instance.
            message (str): The incoming WebSocket message as a string.
        """
        start_time = time.time()

        data = json.loads(message)
        update_id_low = data.get("U")
        update_id_upp = data.get("u")
        if update_id_low is None:
            return
        
        symbol = data.get("s")
        
        #if symbol =='ETHUSDT':
        #    print('m',update_id_upp)
        print(symbol, update_id_low, update_id_upp)
        snapshot_id = self.__lookup_snapshot_id.get(symbol)

        if snapshot_id is None:  ### Check if initial snapshot has been taken - if not: take one and listen for next message
            #print("snapshot_req", symbol)
            self.init_snapshot[symbol] = self.get_snapshot(symbol)
            #print("snapshot received:", symbol, self.__lookup_snapshot_id.get(symbol))
            return
        elif update_id_upp < snapshot_id + 1: ## snapshot taken way too recently - wait for new ones
            #print("retry: snapshot + 1 > update_id_upp")
            return

        #self.__log_message(message)
        prev_update_id = self.__lookup_update_id.get(symbol)

        if prev_update_id is None:
            #print(symbol, update_id_low, snapshot_id, update_id_upp, "(NOTE snapshot_id gets + 1)")

            assert update_id_low   <= snapshot_id + 1 <= update_id_upp
            self.data_queues[symbol].put((self.init_snapshot[symbol], None))
            self.data_queues[symbol].put((message, snapshot_id))
        else:
            assert update_id_low == prev_update_id + 1
            self.data_queues[symbol].put((message, prev_update_id))

        self.__lookup_update_id[symbol] = update_id_upp

        end_time = time.time()  #end timing after message is queued
        #print(f"Message handling time: {end_time - start_time} seconds;", start_time, end_time)
        return
    
    def __process_symbol_data(self, symbol, data_queue, path):
        """
        Processes data for a single symbol, maintaining and serializing the order book.

        Args:
            symbol (str): The trading symbol to process data for.
            data_queue (MPQueue): Queue to receive data updates for the symbol.
        """
        order_book = OrderBookTorch(symbol, custom_stat_fn=self.custom_stat_fn)
        filename = os.path.join(path, f"{symbol}_order_book.bin")
    
        # Open the file locally in the process
        with open(filename, 'wb') as file_handle:
            while True:
                message = data_queue.get()
                start_time = time.time()
                if message is None:
                    break
                data, prev_id = message
                data = json.loads(data)
                snapshot_update_id = data.get("lastUpdateId")
                if snapshot_update_id:
                    order_book.add_snapshot(data, snapshot_update_id)
                else:
                    order_book.add_update(data["u"], prev_id, data["b"], data["a"])
                    #if symbol =='ETHUSDT':
                    #    print(data["u"])
                            #serialize and write after each update
                serialized_data = order_book.serialize()
                file_handle.seek(0)  #beginning of the file
                file_handle.write(serialized_data)
                file_handle.flush()  #ensure data is written to disk
                
                end_time = time.time()  # end timing after message is queued
                #print(f"Storage time: {end_time - start_time} seconds;", start_time, end_time)

    def __load_order_book(self, symbol):
        filename = f"{symbol}_order_book.bin"
        with open(filename, 'rb') as file:
            data = file.read()
            order_book = OrderBookTorch.deserialize(data)
        return order_book

    def __on_error(self, _ws, error):
        print(f"Encountered error: {error}")
        #import traceback
        #traceback.print_exc()
        return

    def __on_close(self, _ws, _close_status_code, _close_msg):
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
        """
        Fetches the initial snapshot for a symbol from the API.

        Args:
            symbol (str): The trading symbol to fetch the snapshot for.

        Returns:
            str: The initial snapshot as a JSON string.
        """
        snapshot_url = f"https://api.binance.us/api/v3/depth?symbol={symbol}&limit=1000"
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

def spread2(bids, asks): ##just a sample custom function
    best_bid_price = torch.max(bids[:, 0])
    best_ask_price = torch.min(asks[:, 0])    
    return 'spread2', (best_ask_price - best_bid_price).item()

def main():
    symbols = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "BNBUSDT", "BNTUSDT"]
    orderbook_client = OrderBookClientGPU(symbols, custom_stat_fn=spread2)
    orderbook_client.start()


if __name__ == '__main__':
    main()
