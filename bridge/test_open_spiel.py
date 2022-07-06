# pylint: enable=line-too-long

import os
import pickle

import haiku as hk
import jax
import numpy as np

from open_spiel_nn import load_model, ob_to_str, str_to_ob

import pyspiel



MIN_ACTION = 52
RANK = '23456789TJQKA'


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


def print_bid_translation(ob):
    hand, bids = ob_to_str(ob)
    print(hand)
    print(bids)

    nob = str_to_ob(hand, bids)
    assert nob == ob


if __name__ == "__main__":
  while True:
    main([])
