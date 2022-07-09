
import os
import sys

from open_spiel_nn import convert_osp_line

import pyspiel
GAME = pyspiel.load_game('bridge(use_double_dummy_result=false)')

def make_dataset(file: str):
  """Creates dataset as a generator of single examples."""
  for line in open(file):
    state = GAME.new_initial_state()
    for action in line.split()[:-52]:
        state.apply_action(int(action))
    #print(state.observation_tensor())
    print(state)
    cards, bids = convert_osp_line(line)
    c_list = ['.'.join(s) for s in cards]
    print(' '.join(c_list))
    print(f"N None {' '.join(bids)}")
    break


def main(argv):
    make_dataset(sys.argv[1])


if __name__ == '__main__':
    main(sys.argv)