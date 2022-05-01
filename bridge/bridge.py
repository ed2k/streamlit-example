#! /usr/bin/env python

# Old Lady
# Copyright(C) 2007 Paul Kuliniewicz <paul@kuliniewicz.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02111-1301, USA.

def _(A): return A
RANKS = range(2, 15)
TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING, ACE = RANKS

SUITS = range(4)
DENOMINATIONS = range(5)
CLUBS, DIAMONDS, HEARTS, SPADES, NO_TRUMP = DENOMINATIONS

PASS = -3
DOUBLE = -2
REDOUBLE = -1

PLAYERS = range(4)
NORTH, EAST, SOUTH, WEST = PLAYERS
PLAYER_NAMES = [_("North"), _("East"), _("South"),_("West")]

TEAMS = range(2)
WEST_EAST, NORTH_SOUTH = TEAMS
TEAM_NAMES = [_("West-East"), _("North-South")]
STR2RANK = {'A':14,'K':13,'Q':12,'J':11,'T':10,'9':9,'8':8,'7':7,'6':6,'5':5,'4':4,'3':3,'2':2}
STR2SUIT = {'S':3,'H':2,'D':1,'C':0}

# S 37+ 2->39 - 51
# H 24+ 26 - (A->38)
# D 11+ 13 - (A->25)
# C -2+ 0 - (A->12)
# S,H,D,C, N,E,S,W
PBN_HIDX = {'2':0,'3':1,'4':2,'5':3,'6':4,'7':5,'8':6,'9':7,'t':8,'j':9,'q':10,'k':11,'a':12}
#E S W N
name_dict = {1:'East',3:'West',2:'South',0:'North'}
IDX = {'E':1,'W':3,'S':2,'N':0}

f_op2argc = {'J':5,'S':4,'C':1,'s':4,'T':4,'e':3,'a':2,'p':2,'Y':3,'*':8}

def f2o(idx):   return idx
def o2f(idx) : return idx
def f2o_card(c): return Card(c // 13, (c % 13)+2)
def o2f_card(c): return c.suit*13 + c.rank-2
def f2o_hand(hand): return [f2o_card(c) for c in hand]
def o2f_hand(hand): return [o2f_card(c) for c in hand]
def pbn2f_card(c):
   ''' cdhs -> 0-3 * 13 + rank'''
   return KIDX[c[0].lower()]*13+PBN_HIDX[c[1].lower()]
def o2pbn_hand(hand):
   '''-> SHDC '''
   h = ['','','','']
   for c in hand:
      h[c.suit] += '23456789TJQKA'[c.rank-2]
   h.reverse()
   return '.'.join(h)
def o2f_bid(b):
   if b.is_pass(): return ' p'
   if b.is_double(): return ' x'
   if b.is_redouble(): return 'xx'
   return str(b.level)+'cdhsn'[b.denom]



import random

class BidGrid:
    ''' n row 4 colum table to record bid N->W 0-3'''
    col = 0
    dealer = 0
    row = 0
    data = None
    def __init__(self, dealer):
        self.data = []
        self.dealer = dealer
        self.data.append([x+10 for x in range(dealer)])
        #print self.data
        self.col = dealer
    def next(self):
        self.col+=1
        if self.col>3:
            self.col = 0
            self.row += 1
    def prev(self):
        self.col-=1
        if self.col < 0:
             self.col = 3
             self.row -= 1
    def end(self):
        self.row = len(self.data)-1
        if self.data[self.row] == []: self.row -= 1
        self.col = len(self.data[self.row])-1
        
    def append(self, bid):
        #print bid, self.row,self.col
        self.data[self.row].append(bid)
        self.col+=1
        if self.col>3:
            self.col = 0
            self.data.append([])
            self.row += 1
    def begin(self):
        self.col = self.dealer
        self.row = 0
    def getDeclarer(self):
        contract, contractor = self.getContract()
        #print 'contract',contract, contractor
        self.begin()
        bid = self.getBid()
        while team(self.col) != team(contractor) or bid is None or bid.denom != contract.denom:
            self.next()
            bid = self.getBid()
        return self.col

    def getContract(self):
        #print 'getcontract',str(self)
        self.end()
        # last three has to be pass
        for i in range(3):
            if not self.getBid().is_pass(): return None
            self.prev()
        while self.getBid().is_pass() or self.getBid().is_double() or self.getBid().is_redouble():
            self.prev()
        return (self.getBid(), self.col)
    def getBid(self):
        #print 'getbid',self.row,self.col
        return self.data[self.row][self.col]

    def __str__(self):
        s = []
        for y in range(len(self.data)):
            s.append(' '.join([str(x) for x in self.data[y]]))
        return '\r\n'.join(s)
                


def denomination_to_string(denom):
    """
    Produces a string representation of a suit or denomination.
    """

    #return ["\xe2\x99\xa3", "\xe2\x99\xa6", "\xe2\x99\xa5", "\xe2\x99\xa0", "NT"][denom]
    return ["c", "d", "h", "s", "n"][denom]


def rank_to_string(rank):
    """
    Produces a string representation of a rank.
    """

    if rank < 10:
        return str(rank)
    else:
        return "TJQKA"[rank - 10]


def team(player):
    """
    Returns the team a player is on.
    """
    return player % 2


def partner(player):
    """
    Returns the partner of a player.
    """

    return (player + 2) % 4

def seat_prev(seat): return (seat-1)%4
def seat_next(seat): return (seat+1)%4
def seat_move(a,b): return (a+b)%4
def seat_str(seat): return 'NESW'[seat]


class Card:
    """
    A single playing card.
    """
    def __init__(self, suit, rank):
        self.suit = suit # 0-3
        self.rank = rank # 2-14

    def __str__(self):
        return rank_to_string(self.rank) + denomination_to_string(self.suit)

    def __lt__(self, other):
        return self.suit < other.suit or self.rank < other.rank

    def __eq__(self, other):
        return self.suit == other.suit and self.rank == other.rank

    def hcp(self):
        if self.rank > 10: return self.rank-10
        return 0
    
    def controls(self):
        if self.rank == 14: # Ace
            return 2
        elif self.rank == 13: # King
            return 1
        return 0

KIDX = {'c':0,'d':1,'h':2,'s':3,'n':4}
def f2o_bid(b):
   #print 'bid',b
   level = 0
   denom = None
   if b == ' p':
      level = PASS
   elif b == ' x':
      level = DOUBLE
   elif b == 'xx':
      level = REDOUBLE
   else:
      level = int(b[0])
      denom = KIDX[b[1]]
   return Bid(level, denom)

class Bid:
    """
    A bid made during the opening phase of a deal.
    """
    def init_str(self, s_bid: str) -> None:
        """ two character form ?c,?d,?h,?s,?n,[ |-]p,[ |-]x,xx
        or do, re
        """
        self.level = PASS
        self.denom = PASS
        if len(s_bid) != 2:
            return
        level = s_bid[0].lower()
        denom = s_bid[1].lower()
        if level in ' -d':
            if denom in 'x' or level in 'd':
                self.level = DOUBLE
                self.denom = DOUBLE
        elif level in 'xr':
            self.level = REDOUBLE
            self.denom = REDOUBLE
        elif level in '1234567':
            if denom in 'cdhsn':
                self.level = int(level)
                self.denom = KIDX[denom]
        

    def __init__(self, level, denom: int=0) -> None:
        if type(level) == str:
            self.init_str(level)
        elif level == DOUBLE:
            self.level = DOUBLE
            self.denom = DOUBLE
        elif level == REDOUBLE:
            self.level = REDOUBLE
            self.denom = REDOUBLE
        elif level in range(1, 8) and denom in range(5):
            self.level = level
            self.denom = denom
        else:
            self.level = PASS
            self.denom = PASS

    def __lt__(self, other):
        if self.is_pass():
            return True
        if other.is_pass():
            return False
        if self.is_double():
            if other.is_double():
                return False
            return True
        if other.is_double():
            return False
        if self.is_redouble():
            if other.is_redouble():
                return False
            return True
        if other.is_redouble():
            return False
        return self.to_num() < other.to_num()
    
    def __ge__(self, other):
        if other.is_pass():
            return True
        if self.is_pass():
            return False
        if other.is_double():
            if self.is_double():
                return False
            return True
        if self.is_double():
            return False
        if other.is_redouble():
            if self.is_redouble():
                return False
            return True
        if self.is_redouble():
            return False
        return not self.__lt__(other)

    def __eq__(self, other):
        return self.level == other.level and self.denom == other.denom

    def __str__(self):
        if self.is_pass():
            return "Pass"[:2]
        elif self.is_double():
            return "Double"[:2]
        elif self.is_redouble():
            return "Redouble"[:2]
        else:
            return str(self.level) + denomination_to_string(self.denom)
    def getIncr(self,i): return Bid(self.level + i, self.denom)
    def getNew(self, denom):
        if self.denom > denom: return Bid(self.level+1, denom)
        return Bid(self.level,denom)
    def getJump(self): return self.getIncr(2)
    def getJumpshift(self, denom):return self.getNew(denom).getIncr(1)
    def is_contract(self):
        """
        Return whether this bid represents a possible contract.
        """

        return self.level > 0

    def is_pass(self):
        """
        Return whether this bid is a pass.
        """

        return self.level == PASS

    def is_double(self):
        """
        Return whether this bid is a double.
        """

        return self.level == DOUBLE

    def is_redouble(self):
        """
        Return whether this bid is a redouble.
        """

        return self.level == REDOUBLE

    def type(self):
        if self.denom == DIAMONDS or self.denom == CLUBS: return str(self.level)+'minor'
        elif self.denom == SPADES or self.denom == HEARTS: return str(self.level)+'major'
        elif self.denom == NO_TRUMP: return 'nt'
        return ''

    def difftype(self, bid):
        '''assume the bid is large then self '''
        if bid.is_pass() or bid.is_double() or bid.is_redouble(): return str(bid)
        if self >= bid: return str(bid)
        d = bid.level - self.level
        n = bid.denom - self.denom
        if n == 0: return '+'+str(d)
        if d == 0: return 'new0'
        if d == 1 and n < 0: return 'new'
        return 'jump'

    def to_num(self):
        if self.is_pass(): return 52
        if self.is_double(): return 53
        if self.is_redouble(): return 54
        return 55 + self.level*5 + self.denom

    @staticmethod
    def next_bid(prev_bids, prev_level_bid=None):
        """
        Return the list of next possibl bid in the sequence.
        prev_bids is a list of previous bids, only need at most last two
        prev_level_bid is the last bid with level and denom, i.e not pass double or redouble
        """
        all_levs = Bid.all_levels()
        if prev_level_bid:
            all_levs = Bid.all_levels(prev_level_bid.level, prev_level_bid.denom)
        if len(prev_bids) == 0:
            return [Bid(PASS)] + all_levs
        
        last_bid = prev_bids[-1]
        if len(prev_bids) == 1:
            if last_bid.is_pass() or last_bid == prev_level_bid:
                return [Bid(PASS), Bid(DOUBLE)] + all_levs
            return [Bid(PASS)] + all_levs
        
        partner_bid = prev_bids[-2]
        if len(prev_bids) == 2:
            if partner_bid.is_pass() and last_bid.is_pass():
                return [Bid(PASS)] + all_levs
            if last_bid == prev_level_bid:
                return [Bid(PASS), Bid(DOUBLE)] + all_levs
            if last_bid.is_double():
                return [Bid(PASS), Bid(REDOUBLE)] + all_levs
            #print('408', bids_str(prev_bids), str(prev_level_bid))
            return [Bid(PASS)] + all_levs

        l3_bid = prev_bids[-3]
        if last_bid.is_pass():
            if partner_bid.is_pass() and l3_bid.is_pass():
                return []
            if partner_bid.is_pass() and l3_bid == prev_level_bid:
                return [Bid(PASS), Bid(DOUBLE)] + all_levs
            if partner_bid.is_pass() and l3_bid.is_double():
                return [Bid(PASS), Bid(REDOUBLE)] + all_levs
        if last_bid.is_double():
            return [Bid(PASS), Bid(REDOUBLE)] + all_levs
        if last_bid == prev_level_bid:
            return [Bid(PASS), Bid(DOUBLE)] + all_levs
        
        return [Bid(PASS)] + all_levs
    
    @staticmethod
    def all_levels(current_level=None, current_denom=None):
        """ current_level and current_denom are optional
        """
        all_levs = [Bid(level, denom) for level in range(1, 8) for denom in range(5)]
        #print('all_levels', current_level, current_denom)
        if not current_level and not current_denom:
            return all_levs
        if current_level == -1 or current_denom == -1:
            return all_levs

        result = [Bid(current_level, denom) for denom in range(current_denom+1, 5)]
        result += [Bid(level, denom) for level in range(current_level+1, 8) for denom in range(current_denom, 5)]
        return result


class Trick:
    """
    A single trick during a deal.
    """

    def __init__(self, player, trump):
        self.player = player
        self.trump = trump
        self.winner = None
        self.cards = [None, None, None, None]
        self.lead = None
        self.leader = player

    def play_card(self, card):
        """
        Play a card during the trick.  Return whether the trick requires
        more cards before determining a winner.
        """

        self.cards[self.player] = card
        if self.lead is None:
            self.lead = card.suit
            self.winner = self.player
        elif (card.suit == self.cards[self.winner].suit and card.rank > self.cards[self.winner].rank) or \
             (card.suit == self.trump and self.cards[self.winner].suit != self.trump):
            self.winner = self.player
        self.player = (self.player + 1) % 4
        return self.cards[self.player] is None
    def __str__(self):
        r = []
        for c in self.cards:
            if c is None: r.append('-')
            else: r.append(str(c))
        return ' '.join(r)


class Deal:
    """
    A single deal of contract bridge.
    """
    def shuffle(self):
        deck = []
        for suit in SUITS:
            for rank in RANKS:
                deck.append(Card(suit, rank))
        random.shuffle(deck)

        self.hands = [deck[0:13], deck[13:26], deck[26:39], deck[39:52]]
        for hand in self.hands:
            hand.sort(reverse = True)
        import defs
        if defs.testing:
            test_deal_gen(self, defs.hands)
            
    def __init__(self, dealer):
        self.played_hands = [[],[],[],[]]
        self.hands = [None, None, None, None]
        self.dealer = dealer
        self.player = dealer
        self.contract = None
        self.passes = 0
        self.contractor = None
        self.doubler = None
        self.redoubler = None

        self.declarer = None
        self.dummy = None
        self.trick = None
        self.opening_lead = False
        self.tricks_taken = [0, 0]

        self.model = BidGrid(self.dealer)

    def legal_bids(self):
        """
        Returns a list of the legal bids available to the next bidder.  If
        bidding is over, the list is empty.
        """

        if (self.contract is not None and self.passes == 3) or self.passes == 4:
            return []

        legal = [Bid('-P')]
        if self.contract is not None:
            for denom in range(self.contract.denom + 1, NO_TRUMP + 1):
                legal.append(Bid(self.contract.level, denom))
            for level in range(self.contract.level + 1, 8):
                for denom in range(NO_TRUMP + 1):
                    legal.append(Bid(level, denom))
        else:
            for level in range(1, 8):
                for denom in range(NO_TRUMP + 1):
                    legal.append(Bid(level, denom))

        if self.contract is not None:
            if self.doubler is None and self.contractor != team(self.player):
                legal.append(Bid(DOUBLE))
            elif self.doubler is not None and self.redoubler is None and self.doubler != team(self.player):
                legal.append(Bid(REDOUBLE))

        return legal

    def bid(self, bid):
        """
        Place the current bidder's bid.  Returns whether bidding continues.
        """

        if bid.is_pass():
            self.passes += 1
        elif bid.is_double():
            self.passes = 0
            self.doubler = team(self.player)
        elif bid.is_redouble():
            self.passes = 0
            self.redoubler = team(self.player)
        else:
            self.contract = bid
            self.passes = 0
            self.contractor = team(self.player)
            self.doubler = None
            self.redoubler = None

        self.model.append(bid)
        self.player = (self.player + 1) % 4

        if self.passes < 3 or (self.passes == 3 and self.contract is None):
            return True
        else:
            #-yisu here we found auction is over,
            self.prepare_trick_taking()
            #TODO get to see the dummy hand for each player, but not now, when?
            return False

    def prepare_trick_taking(self):
        """
        Transition from the bidding phase of the deal to the trick-taking
        phase.
        """
        if self.contract is not None:
            self.declarer = self.model.getDeclarer()
            self.dummy = partner(self.declarer)
            self.player = (self.declarer + 1) % 4
            self.trick = Trick(self.player, self.contract.denom)
            print('dealer',self.dealer,'player',self.player, 'dummy',self.dummy,'declarer',self.declarer,'contract',self.contract)
        else:
            self.contract = Bid('-P')
    def finishBidding(self):
        return self.trick is not None

    def finishTrick(self):
        return self.trick.cards[self.player] is not None

    def legal_card(self, card):
        """
        Check whether a card is legal to play in the current trick.
        """
        return self.trick.lead is None or (self.trick.lead == card.suit) or len([c for c in self.hands[self.player] if c.suit == self.trick.lead]) == 0

    def play_card(self, card):
        """
        Play a card in the current trick.
        """
        mycards = self.hands[self.player]
        if mycards:
            #print(f'try to remove {card} from player {self.player}')
            mycards.remove(card)
            #print_hand(mycards)
            
        self.trick.play_card(card)
        self.played_hands[self.player].append(card)
        #print('hands', [str(c) for c in self.played_hands[self.player]])
        self.player = (self.player + 1) % 4
        # TODO why it is always True?
        self.opening_lead = True
    def originalHand(self, player): return self.hands[player]+self.played_hands[player]        
    def trickCompleted(self):
        return  self.trick.cards[self.player] is not None
    def next_trick(self):
        """
        Advance to the next trick, if there is one.
        """

        self.tricks_taken[team(self.trick.winner)] += 1
        if self.tricks_taken[WEST_EAST] + self.tricks_taken[NORTH_SOUTH] < 13:
            self.player = self.trick.winner
            self.trick = Trick(self.trick.winner, self.contract.denom)
        else:
            self.trick = None

    
class Rubber:
    """
    Keeps dealing hands until one team wins two games; then tallies up points
    to determine who wins.
    """

    def __init__(self, dealer):
        self.deal = None
        self.hands = None
        self.dealer = dealer
        self.above = [0, 0]
        self.below = [0, 0]
        self.vulnerable = [False, False]
        self.rubber = False

    def next_deal(self):
        """
        Start another deal, returning the deal for convenience.
        """

        self.deal = Deal(self.dealer)
        self.deal.shuffle()
        self.dealer = (self.dealer + 1) % 4
        self.hands = [hand[:] for hand in self.deal.hands]

        return self.deal

    def score_deal(self):
        """
        Tally up the scores for the current deal only, and return a list of
        what points were earned and how.
        """

        scoring = []
        level = self.deal.contract.level
        denom = self.deal.contract.denom
        offense = team(self.deal.declarer)
        defense = 1 - offense

        if self.deal.tricks_taken[offense] >= self.deal.contract.level + 6:
            if denom == CLUBS or denom == DIAMONDS:
                points = 20 * level
            elif denom == HEARTS or denom == SPADES:
                points = 30 * level
            else:
                points = 30 * level + 10

            if self.deal.redoubler is not None:
                points *= 4
            elif self.deal.doubler is not None:
                points *= 2

            scoring.append(_("%s earns %d points below for making contract") % (TEAM_NAMES[offense], points))
            self.below[offense] += points

            overtricks = self.deal.tricks_taken[offense] - 6 - level
            if overtricks > 0:
                if self.deal.redoubler is not None and self.vulnerable[offense]:
                    points = 400 * overtricks
                elif self.deal.redoubler is not None:
                    points = 200 * overtricks
                elif self.deal.doubler is not None and self.vulnerable[offense]:
                    points = 200 * overtricks
                elif self.deal.doubler is not None:
                    points = 100 * overtricks
                elif denom == CLUBS or denom == DIAMONDS:
                    points = 20 * overtricks
                else:
                    points = 30 * overtricks
                scoring.append(_("%s earns %d points above for %d overtricks") % (TEAM_NAMES[offense], points, overtricks))
                self.above[offense] += points
                if self.deal.doubler is not None:
                    if self.deal.redoubler is not None:
                        points = 100
                    else:
                        points = 50
                    scoring.append(_("%s earns %d points above for the insult") % (TEAM_NAMES[offense], points))
                    self.above[offense] += points

            if level == 7:
                if self.vulnerable[offense]:
                    points = 1500
                else:
                    points = 1000
                scoring.append(_("%s earns %d points above for the grand slam") % (TEAM_NAMES[offense], points))
                self.above[offense] += points
            elif level == 6:
                if self.vulnerable[offense]:
                    points = 750
                else:
                    points = 500
                scoring.append(_("%s earns %d points above for the slam") % (TEAM_NAMES[offense], points))
                self.above[offense] += points
        else:
            undertricks = 6 + level - self.deal.tricks_taken[offense]
            if self.deal.doubler is not None:
                if self.vulnerable[offense]:
                    points = 200 + 300 * (undertricks - 1)
                else:
                    if undertricks >= 4:
                        points = 100 + 200 + 200 + 300 * (undertricks - 3)
                    elif undertricks >= 2:
                        points = 100 + 200 * (undertricks - 1)
                    else:
                        points = 100
                if self.deal.redoubler is not None:
                    points *= 2
            else:
                if self.vulnerable[offense]:
                    points = 100 * undertricks
                else:
                    points = 50 * undertricks
            scoring.append(_("%s earns %d points above for %d undertricks") % (TEAM_NAMES[defense], points, undertricks))
            self.above[defense] += points

        # TODO: Verify that these points never get (re)doubled.
        for player in PLAYERS:
            if denom == NO_TRUMP:
                aces = [card for card in self.hands[player] if card.rank == ACE]
                if len(aces) == 4:
                    scoring.append(_("%s earns 150 points above for %s having all four aces") % (TEAM_NAMES[team(player)], PLAYER_NAMES[player]))
                    self.above[team(player)] += 150
            else:
                honors = [card for card in self.hands[player] if card.suit == denom and card.rank >= TEN]
                if len(honors) == 5:
                    scoring.append(_("%s earns 150 points above for %s having all five top trump honors") % (TEAM_NAMES[team(player)], PLAYER_NAMES[player]))
                    self.above[team(player)] += 150
                elif len(honors) == 4:
                    scoring.append(_("%s earns 100 points above for %s having four of the five top trump honors") % (TEAM_NAMES[team(player)], PLAYER_NAMES[player]))
                    self.above[team(player)] += 100

        return scoring

    def score_game(self):
        """
        Tally up the scores for the end of the game, if the end of the game
        has indeed been reached.  Returns a list of what points were earned
        and how as a result.
        """

        scoring = []
        offense = team(self.deal.declarer)
        defense = 1 - offense

        if self.below[offense] >= 100:
            scoring.append(_("%s wins a game") % TEAM_NAMES[offense])
            if self.vulnerable[offense]:
                self.rubber = True
                if self.vulnerable[defense]:
                    scoring.append(_("%s earns 500 points above for ending the rubber in three games") % TEAM_NAMES[offense])
                    self.above[offense] += 500
                else:
                    scoring.append(_("%s earns 700 points above for ending the rubber in two games") % TEAM_NAMES[offense])
                    self.above[offense] += 700
            else:
                self.vulnerable[offense] = True
            for pair in TEAMS:
                self.above[pair] += self.below[pair]
                self.below[pair] = 0
            scoring.append(_("Scores below get moved above"))

        return scoring


    def score_rubber(self):
        """
        Tally up the final scores to see who wins the rubber, if the rubber
        has in fact been decided.  Returns a list of what happened.
        """

        scoring = []

        if self.rubber:
            if self.above[WEST_EAST] == self.above[NORTH_SOUTH]:
                scoring.append(_("Both teams tie the rubber!"))
            else:
                if self.above[WEST_EAST] > self.above[NORTH_SOUTH]:
                    victor = WEST_EAST
                else:
                    victor = NORTH_SOUTH
                scoring.append(_("%s wins the rubber!") % TEAM_NAMES[victor])

        return scoring


def print_hand(hand):
    r = []
    for c in hand:
      card = 'SHDC'[c.suit] + '0123456789TJQKA'[c.rank]
      r.append(card)
    #print('ph', r)

def bids_str(bids):
    r = []
    for c in bids:
      card = str(c)
      r.append(card)
    return ''.join(r)


def test_deal_gen(obj, hands):
    import floater_client
    # for debuging biding, use the same deal
    if type(hands) == type(''):
        if len(hands.splitlines()) == 1:
            #pbn = "K8652.Q76.KT8.AK T4.843.65.QT9873 AJ.AJT5.J432.J65 Q973.K92.AQ97.42"
            pbn = hands
            hands = [x.split('.') for x in pbn.split()]
        else:
            s = '''
0 9s Qh Th 2h Kd Jd 7d 6d 5d 3d Ac 5c 3c
1 Js 8s 6s 9h 8h 4h Ad Qd 9d 4d Kc 4c 2c
2 As Ks Qs Ts 2s Ah Kh 3h 8d 2d 9c 8c 6c
3 7s 5s 4s 3s Jh 7h 6h 5h Td Qc Jc Tc 7c
'''            
            hands = hands.splitlines()[1:5]
            for p in PLAYERS:
                h = []
                for c in hands[p].split()[1:]:
                    suit = STR2SUIT[c[1].upper()]
                    card = Card(suit, floater_client.PBN_HIDX[c[0].lower()]+2)
                    h.append(card)
                obj.hands[p] = h
    else:
        example = [
['5', 'KQ98', 'KJ952', 'Q95'],
['K3', 'J7653', '7', 'KT832'],
['AJT9874', 'T4', '3', 'J64'],
['Q62', 'A2', 'AQT864', 'A7'],
]
        #hands = ['KJT43.A6.AK98.Q7', '.QJT32.Q75.AK932', 'A962.754.JT4.T64', 'Q875.K98.632.J85']
        for p in PLAYERS:
            h = []
            suits = hands[p]
            if type(suits) != type([]): suits = suits.split('.')
            for s in SUITS:
                for c in suits[3-s]:
                    card = Card(s,floater_client.PBN_HIDX[c.lower()]+2)
                    h.append(card)
            obj.hands[p] = h    