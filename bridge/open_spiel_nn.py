# pylint: enable=line-too-long

import pickle

import haiku as hk
import jax
import numpy as np

from pathlib import Path
import pyspiel


# Make the network.
NUM_ACTIONS = 38  # pass, double, redouble, 5x7
MIN_ACTION = 52
RANK = '23456789TJQKA'
rRANK = dict(zip(RANK, range(len(RANK))))
PDR = dict(zip('PDR', range(3)))
PXXX = dict(zip(['P', 'X', 'XX'], range(52, 55)))
rPXXX = dict(zip(range(52, 55), ['P', 'X', 'XX']))

def net_fn(x):
  """Haiku module for our network."""
  net = hk.Sequential([
      hk.Linear(1024),
      jax.nn.relu,
      hk.Linear(1024),
      jax.nn.relu,
      hk.Linear(1024),
      jax.nn.relu,
      hk.Linear(1024),
      jax.nn.relu,
      hk.Linear(NUM_ACTIONS),
      jax.nn.log_softmax,
  ])
  return net(x)

def load_model():
  net = hk.without_apply_rng(hk.transform(net_fn))
  f = Path(__file__).parent.joinpath('params-snapshot.pkl')
  params = pickle.load(open(f, 'rb'))
  return net, params


def bid_to_str(bid):
    if bid > 51 and bid < 55:
        return 'PDR'[bid-52]
    b = bid - 55
    return f"{b//5 + 1}{'CDHSN'[b%5]}"

def str_to_bid(bid_s):
    if bid_s in 'PDR':
        return dict(P=52, D=53, R=54)[bid_s]
    rank = int(bid_s[0])
    suit = int(dict(C=0,D=1,H=2,S=3,N=4)[bid_s[1]])
    return 55 + 5 * (rank-1) + suit


def get_legal_actions(auction):
    """ 52, 53, 54 := P, Dbl, RDbl
    55 - 89 ?? 5-suits, 7 bid 1C, 1D, ... 1N, 2C, ... 7N
    auction is serial of bid history in PDR[1-7]CDHSN str format
    return list of open spiel number format 52-89
    """
    # return null if three pass after opening or four pass
    if len(auction) > 3 and all([b == 'P' for b in auction[-4:]]):
        return []
    if len(auction) > 3 and all([b == 'P' for b in auction[-3:]]):
        return []
    legal_auction = [str_to_bid('P')]
    if len(auction) == 0:
        return legal_auction + list(range(55, 90))

    i = len(auction) - 1
    seen_dbl = -1
    seen_rdbl = -1
    while i >= 0:
        if auction[i] not in 'PDR':
            # can't double our own team
            if (len(auction) - i) % 2 != 0 and seen_rdbl == -1 and seen_dbl == -1:
                legal_auction += [str_to_bid('D')]
            if seen_dbl != -1 and (len(auction) - seen_dbl) % 2 != 0:
                legal_auction += [str_to_bid('R')]
            return legal_auction + list(range(str_to_bid(auction[i])+1, 90))
        elif auction[i] == 'D' and seen_rdbl == -1:
            seen_dbl = i
        elif auction[i] == 'R' and seen_dbl == -1:
            seen_rdbl = i
        i -= 1

    return legal_auction + list(range(55, 90))


def ai_action(state, net, params):
  observation = np.array(state.observation_tensor(), np.float32)
  policy = np.exp(net.apply(params, observation))
  probs_actions = [(p, a + MIN_ACTION) for a, p in enumerate(policy) if (a + MIN_ACTION) in state.legal_actions()]
  pred = max(probs_actions)[1]

  _, bids = ob_to_str(state.observation_tensor())
  a = get_legal_actions(bids_to_str(bids))
  if a != state.legal_actions():
    bs = bids_to_str(bids)
    print(bids, bs)
    for b in bs:
        print(str_to_bid(b))
    print(a)
    print(state.legal_actions())
  assert a ==  state.legal_actions()

  if pred not in state.legal_actions():
    print(pred)
    print(state.legal_actions())
    print(probs_actions)
  return pred


def _run_once(state, net, params):
  """Plays bots with each other, returns terminal utility for each player.
  [['2J', 'T74', 'A9K6', '6T47'], ['Q689T', '259A', '75', '92'], ['3A4', 'K6', '834', 'Q358K'], ['75K', 'Q83J', '2TJQ', 'JA']]
  """

  cards = [['', '', '', ''], ['', '', '', ''], ['', '', '', ''], ['', '', '', '']]
  i = 0
  while not state.is_terminal():
    if state.is_chance_node():
      outcomes, probs = zip(*state.chance_outcomes())
      card = np.random.choice(outcomes, p=probs)
      #print(i, 'NESW'[i%4], 'CDHS'[card%4], RANK[card//4])
      cards[i%4][card%4] += RANK[card//4]
      state.apply_action(card)
      i += 1
    else:
      action = state.legal_actions()[0]
      if action > 51:
        print('NESW'[state.current_player()], end=' ')
        print_bid_translation(state.observation_tensor())
      if state.current_player() % 2 == 1:
        # Have simplest play for now
        if action > 51:
          # TODO extend beyond just bidding
          action = ai_action(state, net, params)
        state.apply_action(action)
      else:
        if action > 51:
          action = ai_action(state, net, params)
        state.apply_action(action)
  print(cards)
  return state


def main(argv):
  net, params = load_model()
  results = []
  game = pyspiel.load_game("bridge(use_double_dummy_result=true,dealer_vul=true,non_dealer_vul=true)")

  for i_deal in range(1):
    state = _run_once(game.new_initial_state(), net, params)
    print("Deal #{}; final state:\n{}".format(i_deal, state))
    results.append(state.returns())

  stats = np.array(results)
  mean = np.mean(stats, axis=0)
  stderr = np.std(stats, axis=0, ddof=1) / np.sqrt(1)
  print(u"Absolute score: {:+.1f}\u00b1{:.1f}".format(mean[0], stderr[0]))
  print(u"Relative score: {:+.1f}\u00b1{:.1f}".format(mean[1], stderr[1]))
  return state, results


def bids_to_str(bids):
  """bids: ['3:P', '0:1D', '1D:1D']
  return: ['P', '1D', 'D', 'P', 'P']
  West  North East  South
        Pass  1D    Dbl   
  Pass  Pass  Pass 
  """
  if len(bids) == 0:
    return []
  bids_str = []
  prev_index = 0
  for bid in bids:
    bid_split = bid.split(':')
    b0 = bid_split[0]
    index = int(b0[0])
    if len(bids_str) > 0 and any([b != 'P' for b in bids_str]):
      # add missing pass
      while index != (prev_index + 1) % 4:
        bids_str.append('P')
        prev_index += 1
    if len(b0) > 1:
      bid = b0[1]
    else:
      bid = bid_split[1]
    bids_str.append(bid)
    prev_index = index

  while 0 != (prev_index + 1) % 4:
    bids_str.append('P')
    prev_index += 1
  return bids_str

def str_to_bids(bids_str):
  if len(bids_str) == 0:
    return []
  bids = []
  i =  len(bids_str) - 1
  seat_index = 3
  while i >= 0:
    if bids_str[i] in 'DR':
      bids.insert(0, f'{seat_index}{bids_str[i]}:')
    elif bids_str[i] != 'P':
      bids.insert(0, f'{seat_index}:{bids_str[i]}')  
    else:
      if all([b == 'P' for b in bids_str[:i+1]]):
        bids.insert(0, f'{seat_index}:P')
    seat_index = (seat_index - 1) % 4
    i -= 1

  prev_bid = ''
  i = 0
  while i < len(bids):
    current_bid = bids[i].split(':')[1]
    if len(current_bid) == 0:
      bids[i] += prev_bid
    else:
      prev_bid = current_bid
    i += 1
  return bids

def ob_to_str(ob):
  """ open_spiel observation to string
  current bidder always 0, last bidder always 3
  hand ['46', '6TQ', '468', '359JK']
  bids ['2:P', '3:1H', '0D:1H', '1R:1H']
  """
  hand = [''] * 4
  bids = []
  for i in range(len(ob)):
      if ob[i] > 0.0:
          if i > 431:
                c = i - 432
                print(f"{'CDHS'[c%4]}{RANK[c//4]}", end='')
                hand[c%4] += RANK[c//4]
          elif i > 11:  # 420 7-ranks 5-suits 3-bids (bid, double redouble) 4-seats
                c = i - 12
                bdr = (c%60)%12
                b = f"{bdr%4}:{'1234567'[c//(5*4*3)]}{'CDHSN'[(c%60)//12]}"
                if bdr > 3:
                  b = f"{bdr%4}{'?DR'[bdr//4]}:{b[2:]}" # bdr // 4 > 0
                print(f'{b} ', end='')
                bids.append(b)
          elif i > 7:
                c = i - 8
                print(f"{c%4}:P", end=' ') # who bid pass before opening bid
                bids.append(f'{c%4}:P')
          else:  # 
                print(f'{i:3}', end=' ')
  print(f' {len(ob)}')
  # TODO, rearrange bids in seq for inital pass. ex. '0:P', '1:P', '3:P' -> 301
  bids = str_to_bids(bids_to_str(bids))
  bids_str = bids_to_str(bids)
  nbids = str_to_bids(bids_str)
  
  if nbids != bids:
    print('ERR', bids_str, bids, nbids)
  return hand, bids


def str_to_ob(hand, bids):
  nob = [0.0] * 571
  nob[0] = 1.0
  nob[5] = 1.0
  nob[7] = 1.0
  for bid in bids:
      b0 = bid.split(':')[0]
      if len(b0) == 1:
        b0 = int(b0)
      elif b0[1] == 'D':
        b0 = int(b0[0]) + 4
      else:
        b0 = int(b0[0]) + 8
      b1 = bid.split(':')[1]
      if b1 != 'P':
          offset = 12 + b0 + dict(C=0,D=1,H=2,S=3,N=4)[b1[1]] * 12 + 60 * (int(b1[0])-1)
      else:
          offset = 8 + b0
      nob[offset] = 1.0
      #assert nob[offset] ==  ob[offset]
  for suit in range(4):
      for card in hand[suit]:
          rank = RANK.find(card)
          assert rank > -1 and rank < 14
          # ben uses SHDC 0-3, osp uses CDHS in hand/card
          offset = (3-suit) + rank * 4
          nob[432+offset] = 1.0
          # assert nob[432+offset] ==  ob[432+offset]
  return nob


def data_ben_to_osp(cards_line, bids_line):
  """
  cards: K76.965.Q63.9854 AT83.AJ.J9754.73 Q42.84.T2.AKQT62 J95.KQT732.AK8.J
  [N, E, S, W], seat: S.H.D.C
  bids: N None P P 1C 1H P 1N P 2N P 3D P 3H P 4H P P P
  dealer vulnarability [P|X|XX|...]* P P P
  open spiel card
  """
  cards = ['', '', '', '']
  bids = bids_line.split()
  # osp can only handle N, vul:None, need to rotate seats
  ben_cards = cards_line.split()
  rot_seat = dict(zip('NESW', [0, -1, -2, -3]))
  n = bids[0]
  for i in range(4):
    cards[(i+rot_seat[n])%4] = ben_cards[i]
  result = []
  four_osp = [[], [], [], []]
  for p in range(4):
    suits = cards[p].split('.')
    for s in range(4):
      for c in suits[s]:
        osp_code = rRANK[c] * 4 + (3-s)
        four_osp[p].append(osp_code)
  for i in range(13):
    for p in range(4):
      result.append(four_osp[p][i])
  
  for bid in bids[2:]:
    if bid in ['P', 'X', 'XX']:
      result.append(f"{PXXX[bid]}")
    else:
      r = int(bid[0]) - 1
      s = dict(zip('CDHSN', range(5)))[bid[1]]
      result.append(55+r*5+s)
  return ' '.join([str(n) for n in result])


def data_osp_to_ben(osp_code):
  """a open spiel number from 0-89
  0-51 cards, C2 - SA
  52, 53, 54 := P, Dbl, RDbl
  55 - 89 ?? 5-suits, 7 bid 1C, 1D, ... 1N, 2C, ... 7N
  """
  if osp_code < 52:
    return f"{'CDHS'[osp_code%4]}{RANK[osp_code//4]}"
  if osp_code < 55:
    return f"{rPXXX[osp_code]}"
  c = osp_code - 55
  return f"{c//5+1}{'CDHSN'[c%5]}"

def convert_osp_line(line):
  r = []
  for c in line.split():
    r.append(data_osp_to_ben(int(c)))

  SHDC = dict(zip('SHDC', range(4)))
  cards = [['','','',''], ['','','',''], ['','','',''], ['','','','']]
  for i in range(52):
    c = r[i]
    cards[i%4][SHDC[c[0]]]+=c[1]
  if len(r) > (52+52+3):
    return cards, r[52:-52]
  return cards, r[52:]

def print_bid_translation(ob):
    hand, bids = ob_to_str(ob)
    print(hand)
    print(bids)

    nob = str_to_ob(hand, bids)
    assert nob == ob


if __name__ == "__main__":
  a = "K76.965.Q63.9854 AT83.AJ.J9754.73 Q42.84.T2.AKQT62 J95.KQT732.AK8.J".split()
  b = "P P 1C 1H P 1N P 2N P 3D P 3H P 4H P P P".split()
  osp = data_ben_to_osp(a, b).split()
  osp = '27 46 36 43 18 22 0 20 24 2 40 41 28 16 21 10 42 32 48 47 13 17 3 5 12 25 34 8 29 38 23 4 30 26 35 19 9 44 7 51 14 1 45 15 49 33 11 50 39 31 6 37 52 52 58 59 52 61 52 62 52 64 52 52 52 39 31 3 47 41 13 1 45 21 5 49 17 27 16 7 51 50 18 2 6 15 12 22 23 48 4 24 32 0 20 28 44 25 34 37 9 43 14 26 11 10 30 46 36 33 40 8 29 38 35 19 42'
  print(osp)
  r = []
  for c in osp.split():
    r.append(data_osp_to_ben(int(c)))
  print(r[52:-52])
  SHDC = dict(zip('SHDC', range(4)))
  cards = [['','','',''], ['','','',''], ['','','',''], ['','','','']]
  for i in range(52):
    c = r[i]
    cards[i%4][SHDC[c[0]]]+=c[1]
  print(cards)

  
  while False:
    main([])
