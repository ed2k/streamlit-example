import sys
import argparse
import json
import numpy as np

sys.path.append('../../ben/src')

import conf
from nn.models import Models
from bots import BotBid
from bidding import bidding

from open_spiel_nn import load_model, str_to_bids, str_to_ob, get_legal_actions, bid_to_str

VULN = {
    'None': [False, False],
    'N-S': [True, False],
    'E-W': [False, True],
    'Both': [True, True]
}

def ben_to_str(auction):
    bids = []
    for bid in auction:
        if bid == 'PASS':
            bids.append('P')
        elif bid == 'XX':
            bids.append('R')
        elif bid == 'X':
            bids.append('D')
        else:
            bids.append(bid)
    return bids

def str_to_ben(bid):
    if bid == 'P':
        return 'PASS'
    elif bid == 'R':
        return 'XX'
    elif bid == 'D':
        return 'X'
    else:
        return bid

class MyBid:
    def __init__(self, bid):
        self.bid = bid

MIN_ACTION = 52
class MyModel:
    def __init__(self):
        net, params = load_model()
        self.net = net
        self.params = params

class MyBot:
    def __init__(self, vuln, hand, model):
        if type(model) == type(MyModel()):
            self.bot = model
            self.hand = hand
        else:
            self.bot = BotBid(vuln, hand, model)

    def bid(self, auction):
        if type(self.bot) == type(MyModel):
            return self.bot.bid(auction)
        return self.bot.bit(auction)

    def _get_bid_candidates(self, auction):
        auction = ben_to_str(auction)
        m = self.bot
        hand = self.hand.split('.')
        net = m.net
        params = m.params
        bids = str_to_bids(auction)
        ob = str_to_ob(hand, bids)
        observation = np.array(ob, np.float32)
        policy = np.exp(net.apply(params, observation))
        probs_actions = [(p, a + MIN_ACTION) for a, p in enumerate(policy) if (a + MIN_ACTION) in get_legal_actions(auction)]
        pred = max(probs_actions)[1]
        return [MyBid(str_to_ben(bid_to_str(pred)))]

    def get_bid_candidates(self, auction):
        if type(self.bot) == type(MyModel()):
            return self._get_bid_candidates(auction)
        return self.bot.get_bid_candidates(auction)


def bid_hand(hands, dealer, vuln, models_ns_ew, do_search):
    dealer_i = 'NESW'.index(dealer)
    
    bidder_bots = [MyBot(VULN[vuln], hand, models_ns_ew[i % 2]) for i, hand in enumerate(hands)]

    auction = ['PAD_START'] * dealer_i

    turn_i = dealer_i

    while not bidding.auction_over(auction):
        if do_search:
            bid = bidder_bots[turn_i].bid(auction).bid
            auction.append(bid)
        else:
            candidates = bidder_bots[turn_i].get_bid_candidates(auction)
            bid = candidates[0].bid
            auction.append(bid)
        
        turn_i = (turn_i + 1) % 4  # next player's turn
    
    return auction

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--bidderNS', type=str)
    parser.add_argument('--bidderEW', type=str)
    parser.add_argument('--set', type=str)
    parser.add_argument('--search', action='store_true')

    args = parser.parse_args()
    
    sys.stderr.write(f'NS = {args.bidderNS}\n')
    sys.stderr.write(f'EW = {args.bidderEW}\n')
    sys.stderr.write(f'search = {args.search}\n')
    
    if args.bidderNS:
        models_ns = Models.from_conf(conf.load(args.bidderNS))
    else:
        models_ns = MyModel()
    if args.bidderEW:
        models_ew = Models.from_conf(conf.load(args.bidderEW))
    else:
        models_ew = MyModel()

    for line in open(args.set):
        parts = line.strip().split()
        dealer = parts[0]
        # for simplicity always assume North opening
        dealer = 'N'
        vuln = parts[1]
        # open_spiel can only handle None
        vuln = 'None'
        hands = parts[2:]

        auction = bid_hand(hands, dealer, vuln, [models_ns, models_ew], args.search)

        record = {
            'dealer': dealer,
            'vuln': vuln,
            'north': hands[0],
            'east': hands[1],
            'south': hands[2],
            'west': hands[3],
            'auction': auction,
            'contract': bidding.get_contract(auction)
        }

        print(json.dumps(record))
        sys.stdout.flush()
