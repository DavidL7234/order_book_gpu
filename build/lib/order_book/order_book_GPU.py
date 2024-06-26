import torch
import pickle
import time

class OrderBookTorch:
    """
    A class for managing an order book for a specific trading symbol using PyTorch tensors to optimize performance.

    Attributes:
        symbol (str): The trading symbol for the order book.
        max_entries (int): Maximum number of entries the order book can hold.
        custom_stat_fn (callable): A custom function for calculating additional statistics.
        frames (dict): A dictionary to store the snapshots and updates of the order book.
        order_flow (dict): A dictionary tracking the count of different types of orders.
    """

    def __init__(self, symbol, max_entries=1000, custom_stat_fn=None):
        """
        Initializes the OrderBookTorch object.

        Args:
            symbol (str): The trading symbol for the order book.
            max_entries (int): Maximum number of entries the order book can hold, default is 1000.
            custom_stat_fn (callable): Optional function for calculating additional custom statistics.
        """

        self.symbol = symbol
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print("Order Book Device:", self.device)
        self.max_entries = max_entries
        self.frames = {}

        self.order_flow = {
            'buy_orders': 0,
            'sell_orders': 0,
            'cancel_buy_orders': 0,
            'cancel_sell_orders': 0,
            'modify_buy_orders': 0,
            'modify_sell_orders': 0
        }
        self.custom_stat_fn = custom_stat_fn

    def add_snapshot(self, snapshot, update_id):
        """
        Adds a new snapshot to the order book.

        Args:
            snapshot (dict): Contains bids and asks to be added.
            update_id (int): Unique identifier for the update.
        """
                
        bids = [[float(x) for x in bid] for bid in snapshot.get('bids')]
        asks = [[float(x) for x in ask] for ask in snapshot.get('asks')]

        bids = torch.tensor(bids, dtype=torch.float32, device=self.device)
        asks = torch.tensor(asks, dtype=torch.float32, device=self.device)


        start = time.time()
        if bids.size(0) > 0 and asks.size(0) > 0:
            self.frames[update_id] = {
                'bids': bids[:, :2],  
                'asks': asks[:, :2],
                'time': 0,#snapshot["E"],
                'stats': self.compute_statistics(bids, asks)
            }
        print(time.time() - start)

    def add_update(self, update_id, prev_id, bids, asks, event_time):
        """
        Processes an update to the order book.

        Args:
            update_id (int): Unique identifier for the update.
            prev_id (int): Identifier of the previous snapshot to base the update on.
            bids (list): List of bid entries.
            asks (list): List of ask entries.
        """
        # Only process if previous snapshot exists.
        if prev_id in self.frames:
            prev_bids = self.frames[prev_id]['bids']
            prev_asks = self.frames[prev_id]['asks']

            bids = [[float(x) for x in bid] for bid in bids]
            asks = [[float(x) for x in ask] for ask in asks]

            updated_bids, new_buy_orders, cancel_buy_orders, modify_buy_orders = self.update_orders(prev_bids, bids, 'buy')
            updated_asks, new_sell_orders, cancel_sell_orders, modify_sell_orders = self.update_orders(prev_asks, asks, 'sell')

            #start = time.time()
            self.frames[update_id] = {
                'bids': updated_bids,
                'asks': updated_asks,
                'time': event_time,
                'stats': self.compute_statistics(updated_bids, updated_asks)
            }
            #print(time.time() - start)

            self.order_flow['buy_orders'] += new_buy_orders
            self.order_flow['sell_orders'] += new_sell_orders
            self.order_flow['cancel_buy_orders'] += cancel_buy_orders
            self.order_flow['cancel_sell_orders'] += cancel_sell_orders
            self.order_flow['modify_buy_orders'] += modify_buy_orders
            self.order_flow['modify_sell_orders'] += modify_sell_orders


    def update_orders(self, existing_orders, new_orders, order_type):
        """
        Updates the order book entries based on new order data.

        Args:
            existing_orders (torch.Tensor): The current order entries in the order book.
            new_orders (list): New orders to be integrated.
            order_type (str): Type of the order ('buy' or 'sell').

        Returns:
            tuple: Updated orders, count of new, cancelled, and modified orders.
        """
        new_orders_tensor = torch.tensor(new_orders, dtype=torch.float32, device=self.device)

        new_orders_count = 0
        cancel_orders_count = 0
        modify_orders_count = 0


        for new_order in new_orders_tensor:
            price = new_order[0]
            volume = new_order[1]
            idx = (existing_orders[:, 0] == price).nonzero(as_tuple=True)[0]
            if idx.nelement() > 0:
                if volume == 0:
                    cancel_orders_count += 1
                else:
                    existing_orders[idx, 1] = volume
                    modify_orders_count += 1
            else:
                if volume > 0:
                    zero_idx = (existing_orders[:, 1] == 0).nonzero(as_tuple=True)[0]
                    if zero_idx.nelement() > 0:
                        existing_orders[zero_idx[0]] = new_order
                    else:
                        existing_orders = torch.cat((existing_orders, new_order.unsqueeze(0)), dim=0)
                    new_orders_count += 1
        return existing_orders, new_orders_count, cancel_orders_count, modify_orders_count

    
    def compute_statistics(self, bids, asks):
        """
        Computes various statistics from the current state of the bids and asks.

        Args:
            bids (torch.Tensor): Tensor of bid orders.
            asks (torch.Tensor): Tensor of ask orders.

        Returns:
            dict: Dictionary containing computed statistics.
        """
        stats = {}
        best_bid_price = torch.max(bids[:, 0])
        best_ask_price = torch.min(asks[:, 0])
        
        stats['bid_ask_spread'] = (best_ask_price - best_bid_price).item()
        stats['depth_bids'] = torch.sum(bids[:, 1]).item()
        stats['depth_asks'] = torch.sum(asks[:, 1]).item()
        
        best_bid_volume = bids[bids[:, 0] == best_bid_price][:, 1].sum()
        best_ask_volume = asks[asks[:, 0] == best_ask_price][:, 1].sum()
        
        stats['volume_at_best_bid'] = best_bid_volume.item()
        stats['volume_at_best_ask'] = best_ask_volume.item()
        
        stats['market_imbalance'] = (best_bid_volume - best_ask_volume).item()
        stats['quote_volume'] = bids.size(0) + asks.size(0)
        
        if bids.size(0) > 1:
            bid_price_diff = bids[1:, 0] - bids[:-1, 0]
            bid_volume_diff = bids[1:, 1] - bids[:-1, 1]
            slope_bids = torch.abs(bid_volume_diff / bid_price_diff).mean()
        else:
            slope_bids = torch.tensor(0.0, device=self.device)
            
        if asks.size(0) > 1:
            ask_price_diff = asks[1:, 0] - asks[:-1, 0]
            ask_volume_diff = asks[1:, 1] - asks[:-1, 1]
            slope_asks = torch.abs(ask_volume_diff / ask_price_diff).mean()
        else:
            slope_asks = torch.tensor(0.0, device=self.device)
        
        stats['order_book_slope_bids'] = slope_bids.item()
        stats['order_book_slope_asks'] = slope_asks.item()

        stats['order_flow'] = self.get_order_flow()

        if self.custom_stat_fn:
            key,value = self.custom_stat_fn(bids, asks)
            stats[key] = value
        
        return stats

    def get_order_flow(self):
        """
        Calculates the net order flow based on the order activities.

        Returns:
            int: Net order flow value.
        """
        net_order_flow = (self.order_flow['buy_orders'] - self.order_flow['cancel_buy_orders']) - \
                         (self.order_flow['sell_orders'] - self.order_flow['cancel_sell_orders'])
        return net_order_flow
    
    def serialize(self):
        """
        Serializes the entire order book for storage or transmission.

        Returns:
            bytes: Serialized byte stream of the order book.
        """
        saved_data = {update_id: {
            'bids': frame['bids'].cpu().numpy(),
            'asks': frame['asks'].cpu().numpy(),
            'time': frame['time'],
            'stats': frame['stats']
        } for update_id, frame in self.frames.items()}
        return pickle.dumps(saved_data)

    @staticmethod
    def deserialize(data):
        """
        Deserializes the byte stream back into an OrderBookTorch object.

        Args:
            data (bytes): Serialized byte stream of an order book.

        Returns:
            OrderBookTorch: A new instance of OrderBookTorch loaded with the deserialized data.
        """
        loaded_data = pickle.loads(data)
        new_order_book = OrderBookTorch("SymbolName")
        new_order_book.frames = {update_id: {
            'bids': torch.tensor(frame['bids'], device=new_order_book.device),
            'asks': torch.tensor(frame['asks'], device=new_order_book.device),
            'time': frame['time'],
            'stats': frame['stats']
        } for update_id, frame in loaded_data.items()}
        return new_order_book
