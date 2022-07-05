# pylint: enable=line-too-long

import os
import pickle

import haiku as hk
import jax
import numpy as np

import pyspiel


# Make the network.
NUM_ACTIONS = 38
MIN_ACTION = 52
RANK = '23456789TJQKA'

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
  params = pickle.load(open(os.path.join('bridge', 'params-snapshot.pkl'), 'rb'))
  return net, params

def ai_action(state, net, params):
  observation = np.array(state.observation_tensor(), np.float32)
  policy = np.exp(net.apply(params, observation))
  probs_actions = [(p, a + MIN_ACTION) for a, p in enumerate(policy) if (a + MIN_ACTION) in state.legal_actions()]
  pred = max(probs_actions)[1]
  if pred not in state.legal_actions():
    print(pred)
    print(state.legal_actions())
    print(probs_actions)
  return pred


def _run_once(state, bots, net, params):
  """Plays bots with each other, returns terminal utility for each player."""
  for bot in bots:
    bot.restart()
  
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
  bots = []

  results = []
  game = pyspiel.load_game("bridge(use_double_dummy_result=true,dealer_vul=true,non_dealer_vul=true)")

  for i_deal in range(1):
    state = _run_once(game.new_initial_state(), bots, net, params)
    print("Deal #{}; final state:\n{}".format(i_deal, state))
    results.append(state.returns())

  stats = np.array(results)
  mean = np.mean(stats, axis=0)
  stderr = np.std(stats, axis=0, ddof=1) / np.sqrt(1)
  print(u"Absolute score: {:+.1f}\u00b1{:.1f}".format(mean[0], stderr[0]))
  print(u"Relative score: {:+.1f}\u00b1{:.1f}".format(mean[1], stderr[1]))
  return state, results


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
          elif i > 11:  # 420 7 ranks 5 suits 4 seats 3 bids (bid, double redouble)
                c = i - 12
                bdr = (c%60)%12
                b = f"{bdr%4}:{'1234567'[c//(5*4*3)]}{'CDHSN'[(c%60)//12]}"
                if bdr > 3:
                  b = f"{bdr%4}{'?DR'[bdr//4]}:{b[2:]}" # bdd // 4 > 0
                print(f'{b} ', end='')
                bids.append(b)
          elif i > 7:
                c = i - 8
                print(f"{c%4}:P", end=' ') # who bid pass before opening bid
                bids.append(f'{c%4}:P')
          else:  # 
                print(f'{i:3}', end=' ')
  print(f' {len(ob)}')
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
          offset = suit + rank * 4
          nob[432+offset] = 1.0
          # assert nob[432+offset] ==  ob[432+offset]
  return nob


def print_bid_translation(ob):
    hand, bids = ob_to_str(ob)
    print(hand)
    print(bids)

    nob = str_to_ob(hand, bids)
    assert nob == ob


if __name__ == "__main__":
  while True:
    main([])
