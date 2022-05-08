# pylint: enable=line-too-long

import os
import pickle
import re
import time

import numpy as np

import haiku as hk
import jax
import numpy as np

import pyspiel

# FLAGS = flags.FLAGS
# flags.DEFINE_integer("num_deals", 6, "How many deals to play")
# flags.DEFINE_integer("sleep", 0, "How many seconds to wait before next action")
# flags.DEFINE_string("params_path", 'bridge', "directory path for trained model params-snapshot.pkl")

# Make the network.
NUM_ACTIONS = 38
MIN_ACTION = 52

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
  while not state.is_terminal():
    if state.is_chance_node():
      outcomes, probs = zip(*state.chance_outcomes())
      state.apply_action(np.random.choice(outcomes, p=probs))
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


def print_bid_translation(ob):
    hand = [[],[],[],[]]
    bids = []
    for i in range(len(ob)):
        if ob[i] > 0.0:
            if i > 431:
                  c = i - 432
                  print(f"{'CDHS'[c%4]}{'23456789TJQKA'[c//4]}", end='')
                  hand[c%4].append('23456789TJQKA'[c//4])
            elif i > 11:  # 420 7 ranks 5 suits 4 seats 3 bids (bid, double redouble)
                  c = i - 12
                  print(f"{(c%60)%12}:{'1234567'[c//(5*4*3)]}{'CDHSN'[(c%60)//12]}", end=' ')
                  bids.append(f"{(c%60)%12}:{'1234567'[c//(5*4*3)]}{'CDHSN'[(c%60)//12]}")
            elif i > 7:
                  c = i - 8
                  print(f"{c%4}:P", end=' ') # who bid pass before opening bid
                  bids.append(f'{c%4}:P')
            else:  # 
                  print(f'{i:3}', end=' ')
    print()
    print(hand)
    print(bids)
    nob = [0.0] * 571
    nob[0] = 1.0
    nob[5] = 1.0
    nob[7] = 1.0
    for bid in bids:
        b0 = int(bid.split(':')[0])
        b1 = bid.split(':')[1]
        if b1 != 'P':
            offset = 12 + b0 + dict(C=0,D=1,H=2,S=3,N=4)[b1[1]] * 12 + 60 * (int(b1[0])-1)
        else:
            offset = 8 + b0
        nob[offset] = 1.0
        assert nob[offset] ==  ob[offset]
    for suit in range(4):
        for card in hand[suit]:
            rank = '23456789TJQKA'.find(card)
            assert rank > -1 and rank < 14
            offset = suit + rank * 4
            nob[432+offset] = 1.0
            assert nob[432+offset] ==  ob[432+offset]
    assert nob == ob


# if __name__ == "__main__":
#   main([])
