
import os
import sys

from open_spiel_nn import data_ben_to_osp, convert_osp_line


def test(line0, line, nline):
    """ test using open_spiel"""
    import pyspiel
    GAME = pyspiel.load_game('bridge(use_double_dummy_result=false)')
    state = GAME.new_initial_state()
    for action in nline.split():
        state.apply_action(int(action))
    print(state)
    cards, bids = convert_osp_line(nline)
    c_list = ['.'.join(s) for s in cards]
    if line0[:-1] != ' '.join(c_list):
        print('ERR', [line0], c_list)
    if line.split()[2:] != bids:
        print('ERR', line, bids)


def make_dataset(file: str):
  """ Converts a ben file to an osp file.
  732.7.9765.QJ432 AKT6.AJT.AJ42.98 4.K98542.T3.AKT6 QJ985.Q63.KQ8.75
  S E-W 1H 1S P 2H P 2S P 4S P P P
  """
  i = 0
  card = ''
  for line in open(file):
    if i % 2 == 0:
        card = line
    if i % 2 == 1:
        nline = data_ben_to_osp(card, line)
        print(nline)
    i += 1


def main(argv):
    if len(sys.argv) == 2:
        return make_dataset(sys.argv[1])
    card = '732.7.9765.QJ432 AKT6.AJT.AJ42.98 4.K98542.T3.AKT6 QJ985.Q63.KQ8.75'
    line = 'S E-W 1H 1S P 2H P 2S P 4S P P P'
    nline = data_ben_to_osp(card, line)
    print(nline)
    test(card, line, nline)
    

if __name__ == '__main__':
    main(sys.argv)