from collections import namedtuple
import pandas as pd
import streamlit as st

import sys
sys.path.append('bridge')
import audit
import nnet

import test_open_spiel

s, r = test_open_spiel.main([])
st.text(s)
#st.text(r)

action = nnet.main([])
st.write(action)

d = audit.parse_sequence('bridge/SequenceFile.txt')
cnvts = audit.parse_convention('bridge/Std American', d.keys())
cnvts.append('STD')
cnvts_list = []
for cnvt in cnvts:
    for line in d[cnvt]:        
        cnvts_list.append(line)

bids = st.text_input("Standard american biding convention", "1D 2D")
st.write(f"The meaning of {bids} is")

lines = []
for rule in audit.explain_bid(bids, cnvts_list):
    lines.append(rule)
st.text('\n'.join(lines))

row = [st.columns(3), st.columns(3), st.columns(3)]

for i in range(3):
    for j in range(3):
        with row[i][j]:
            b = 'AKQJT98765432'[i*3+j]
            if st.button(b):
                st.write(b)


