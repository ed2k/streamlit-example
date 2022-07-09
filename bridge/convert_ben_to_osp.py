
import os
import sys

from open_spiel_nn import data_ben_to_osp, convert_osp_line

import pyspiel
GAME = pyspiel.load_game('bridge(use_double_dummy_result=false)')

def make_dataset(file: str):
  """ Converts a ben file to an osp file. """
  i = 0
  card = ''
  for line in open(file):
    if i % 2 == 0:
        line0 = line
        card = line
    if i % 2 == 1:
        bids = line.split()
        #print(card, bids)
        nline = data_ben_to_osp(card, line)
        print(nline)
        # test using open_spiel
        # state = GAME.new_initial_state()
        # for action in nline.split():
        #     state.apply_action(int(action))
        # #print(state)
        # cards, bids = convert_osp_line(nline)
        # c_list = ['.'.join(s) for s in cards]
        # if line0[:-1] != ' '.join(c_list):
        #     print('ERR', [line0], c_list)
        # if line.split()[2:] != bids:
        #     print('ERR', line, bids)
    i += 1


def main(argv):
    make_dataset(sys.argv[1])


if __name__ == '__main__':
    main(sys.argv)