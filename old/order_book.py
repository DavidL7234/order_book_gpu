import pickle

class OrderBook:
    def __init__(self, symbol):
        self.symbol = symbol
        self.frames = {}

    def add_snapshot(self, snapshot):
        update_id = snapshot.get('lastUpdateId')
        if update_id:
            self.frames[update_id] = {
                'bids': {price: amount for price, amount in snapshot.get('bids')},
                'asks': {price: amount for price, amount in snapshot.get('asks')},
            }

    def add_update(self, message, prev_id):
        update_id = message.get('u')
        if update_id:
            prev_frame = self.frames.get(prev_id)

            new_bids = {price: amount for price, amount in message.get('b')}
            new_asks = {price: amount for price, amount in message.get('a')}
            
            for b_p,a in new_bids.items():
                prev_frame['bids'][b_p] = a

            for b_a, a in new_asks.items():
                prev_frame['asks'][b_a] = a

            updated_bids = {price: amount for price, amount in prev_frame['bids'].items() if amount != 0}
            updated_asks = {price: amount for price, amount in prev_frame['asks'].items() if amount != 0}


            self.frames[update_id] = {'bids': updated_bids, 'asks': updated_asks}

    def get_frame(self, update_id):
        return self.frames.get(update_id)

    def serialize(self):
        return pickle.dumps(self)

    @staticmethod
    def deserialize(data):
        return pickle.loads(data)
