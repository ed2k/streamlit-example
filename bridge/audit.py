import re

from bridge import Bid

CONVENTION_HIERACHY = """ convention symbols hierarchy
CC=>GENS 1NT 2and3NT Majors Minors 2C2D 2H2S Other doubles NTovercalls overcalls def.vsNT preempts overT/ODbl cueBids Vs.Preempts Slam suitLeads NTleads carding

GENS=STD  2/1  K-S  PREC  GOREN  SIF (CC:General Approach)

1NT=ONAH ONAL NTNV NTVU NT12 NT34 NTTheyV NTTheyNV M:1nopts 2:1nopts F:1nopts (Main, 2nd, forcing club-1DorX?-1N)
ONAH=0..40 (1NT: Main 1NT HCP range high)
ONAL=0..40 (1NT: Main 1NT HCP range low)
ONBH=0..40 (1NT: 2nd 1NT HCP range high)
ONBL=0..40 (1NT: 2nd 1NT HCP range low)
NTNV=(2nd NT Non-vulnurable, vulnerable)
NTVU=(2nd NT vulnerable)
NT12=(2nd NT 1st/2nd seat)
NT34=(2nd NT 3rd/4th seat)
NTTheyV=(2nd NT they vulnerable)
NTTheyNV=(2nd NT they non-vulnurable)
1nopts=n2C 1N2N nAttm 1N2D 1N3C 1N3D 1N2H 1N3H 1N3S 1N2S nOther
n2C=Stay Pup1 GARB (stayman, puppet, garbage stayman)
1N2N=>INV  3C  3D (1nopts:invitational, 3C transfer, 3D transfer)
nAttm=RelayGood RelayPoor (1nopts: after transfer to minor cheapest bid shows relay good, relay poor support)
1N2D=>NAT  JACH  CART (1nopts:natural, jacoby to 2H, Forcing Stayman)
1N2H=>NAT  JACS (1nopts:natural, jacoby to 2S)
1N2S=>NAT  3C  MSS Either (1nopts:natural, 3C transfer, MSS, either support)
1N3C=>F  I  W (1nopts:Forcing, Invitational, Weak)
1N3D=>F  I  W (1nopts:Forcing, Invitational, Weak)
1N3H=>F  I  W 55I (1nopts:Forcing, Invitational, Weak,5-5 inv)
1N3S=>F  I  W 55F (1nopts:Forcing, Invitational, Weak, 5-5 Mjors Game Forcing)
nOther=SYSON TEX1 LEBF LEBS 1N-X ACCP WALS XX2S BELA(System on over All,Texas transfers,smolen,Lebensohl,Leb slow denies, Neg. doubles, super acceptances walsh relays,redboule is tfx to 2C,belladonna relays)

2and3NT=TNH TNL 2N3C HNH HNL 2Other 2N3S 3N
TNH=21
TNL=20
HNH=27
HNL=25
2Other=JAC2 TEX2 SMOL2
2N3C=>STAY  PUP2  BARON (2N:stayman, puppet stayman, baron)
2N3S=>4C  MSS  MIN (2N:transfer to clubs, Minor Suit stayman, both minors)
3N=>BAL  GAMB  NAMY (3N:balanced, gambling NT, nmyats)

Majors=Len12 Len34 1M3M 1MOV3M 1M1N CONSTR SPLINT BERGEN REVB 1M3N SWISS 1MQ 1M2N DRU M2NH M2NL M3NH M3NL(: major suit opening bids)
Len12=4MAJ12 5MAJ12 Can12(Majors:4Major 1st or 2nd hand,5Major12,Canape12)
Len34=4MAJ34 5MAJ34 Can34(Majors:4Major 3rd or 4th hand,5Major34,Canape34)
1M3M=>F  I  W (Majors:double raise forcing, invitational, weak)
1MOV3M=>F  I  W (Majors:double raise after overcall 1M-3M in comp.)
1M1N=>F  S  N (Majors:1NT response forcing 6-13, semi-forcing 6-12,non-forcing 6-10)
CONSTR=(Majors:Constructive Major Raise)
SPLINT=(Majors:Splinter Bids)
BERGEN=(Majors:Bergen Raises)
REVB=(Majors:reverse bergen)
1M3N=(Majors:3NT is conventional raise of 1Major)
SWISS=Inverted Trump Honor Swiss raises (Majors:)
1MQ=>WEST  LR  GF (Majors:Cue bid is westen when we open a Major, limit raise+, game force)
1M2N=>BAL  JAC (Majors:balanced or Jacoby 2NT raise)
DRU=>NO  REG  REV  2WAY (Majors:drury reverse, 2way, regular, no-trump)
M2NH=15 (Majors:2N HCP range high)
M2NL=13
M3NH=18
M3NL=16

Minors=CLEN DLEN 1C3C 1CO3C 1CFR FREQ 1C2NI 1CQ C2NH C2NL C3NH C3NL(: minor suit opening bids)
CLEN=>4  3  2  ? (Minors:exp. len of clubs)
DLEN=>4  3  2  ? (Minors:exp. length of diamonds)
1C3C=>F  I  W (Minors:1min - 3min)
1CO3C=>F  I  W (Minors:1min-3min in comp.)
1CFR=>INVM  OTHM  SPLINTER NONE (Minors:forcing single raise, j/s other min, splinters, none)
FREQ=(Minors:Frequently bypass 4 card diamond suit)
1C2NI=F  I(Minors:1minor-2NT is Invitational,forcing)
1CQ=>WEST  LR  GF  (Minors:Q-bid after open a minor)
C2NH=12 (Minors:2N response HCP range high)
C2NL=11
C3NH=15
C3NL=13

2C2D=TCHH TCHL TDHH TDHL 2C 2ndNeg 2D2N 2D 2DNSF 2CR
TCHH=37 (2C2D:2C HCP range high)
TCHL=22
TDHH=10 (2C2D:2D HCP range high)
TDHL=6_ (2C2D:underscore means?)
2C=>F  PREC  FC (2C2D:forcing with clubs, precision 2C, forcing 2C)
2ndNeg=(2C2D:Cheapest minor used as 2nd negative after 2C-2D)
2D2N=>NEG  FEAT  OG  ASK(2C2D:negative,feature,ogust,asking)
2D=>WK  FLAN  ROMA  PREC  FD(2C2D:weak, flanner, roman, precision, forcing double)
2DNSF=>Y  N  (2C2D:after wk 2-new suit forcing)
2CR=>NAT  WAIT  NEG  CTLS  ACES Albarran(2C2D:response 2H double negative, 2d waiting/asking, 2d negative, control, number of aces, albarran)

2H2S=THHH THHL TSHH TSHL 2H 2S 2H2N 2S2N 2SNSF 2HNSF
THHH=9_ (2H2S:2H HCP range high)
THHL=6_
TSHH=8_ (2H2S:2S HCP range high)
TSHL=6_
2HNSF=>Y  N  (2H2S:2H-new suit forcing)
2H=>WK  FH (2H2S:forcing with hearts)
2H2N=>NEG  NAT  FEA  OG  ASK(2H2S:negative, natural, feature, ogust, flan'ry)
2S=>WK  FS (2H2S:forcing with spades)
2S2N=>NEG  NAT  FEA  OG ASK(2H2S:)
2SNSF=>Y N (2H2S:2S-new suit forcing)

Other=after1NT WJSA WJSC FSF1 FSFG UNUN CRASH BROMAD PASK BART ITAL WITTES  (CC:Other conventional bids ...)
after1NT=...(Other: converntions after 1NT rebid standard, new minor forcing, checkbac stayman, 2-way checkback)
WJSA=Weak Jump Shifts - Always (Other:)
WJSC=Weak Jump Shifts - only in comp.(Other:)
FSF1=4th Suit Forcing - 1 round(Other:)
FSFG=4th Suit Forcing - to game(Other:)
UNUN=Unusual vs. Unusual(Other:)
CRASH=(Other:)
BROMAD=(Other:)
PASK=(Other:)
BART=(Other:)
ITAL=(Other:)
WITTES=(Other:)

doubles=NEGX RESX SUPX MAXX ROSX XPEN ACTX XOFF
NEGX=>3S  (doubles:Neg. X thru)
RESX=>3S  (dboules:Resp. X thru)
SUPX=>3S (doubles:Sup. X thru)
MAXX
ROSX
XPEN
ACTX
XOFF

NTovercalls=D:nopts  B:nopts  DIRNTNAT OVNH OVNL BALNTNAT BALH BALL UNNT SAND(Direct NT, balancing)
(D Direct NT nopts)
DIRNTNAT=Direct NT natural
OVNH=18 (NTovercalls:direct 1NT overcall HCP range high)
OVNL=15
(B balancing nopts)
BALNTNAT=Balancing NT natural
BALH=14 (NTovercalls:balancing 1NT HCP range high)
BALL=10
UNNT=>MIN  LOW  NAT (NTovercalls:wNT o'call shows both minors,2 lowest unbid, natural)
SAND=Sandwich NT(NTovercalls:)

overcalls=SOCH SOCL OVC4 OVCL NEWS JR JQLR JO LOTT
SOCH=17 (overcalls:stimple overcall HCP range high)
SOCL=8_
OVC4=(overcalls:1 level o'call often withi only 4 cards)
OVCL=(overcalls:1 level very light overcalls)
NEWS=>F  I  W  TA(overcalls:response of New suit forcing after o'call, invitational, weak, txf adv)
JR=>F  I  W (overcalls:Jump raise after overcall)
JQLR=(overcalls:Jump Q  limited raise after we overcall
JO=>S  I  W (overcalls:Jump o'call is Strong, Inter, Weak)
LOTT=(overcalls:in competitive bidding situatios often base bids on Law Of Total Tricks)

def.vsNT=VNTW VNTS VNTSBAL VNTWBAL STRONGNTHCP Natural Astro Brozel Cansino Capp/Ham DONT HELLO Landy Meckwell Wooslwel(defense vs. NT)
VNTW=>NAT..WOOL(def.vsNT:vs. weak NT natural,astro,brozel,cansino,capp/ham,DONT,Hello,Landy,Meckwell,Wooslsey)
VNTS=>NAT AST BRO CAN CAPP DONT HELLO LAN WOOL (def.vsNT:vs. strong NT)
VNTSBAL=NAT..WOOL(def.vsNT:vs. strong NT balanced)
VNTWBAL=NAT..WOOL(def.vsNT:vs. weak NT balanced)
STRONGNTHCP=12  13  14  15(def.vsNT:strong NT HCP range)
Natural=(DBL Penalty, 2C long C, 2D long D, 2H long H, 2S long S)
Astro=(DBL Penalty, 2C H+minor, 2D S+minor, 2H long H, 2S long S)
Brozel=(DBL 1 long suit, 2C C+H, 2D D+H, 2H H+S, 2S S+minor)
Cansino=(DBL penalty, 2C C+2suits, 2D H+S, 2H long H, 2S long S)
Capp/Ham=(DBL penalty, 2C 1 long suit, 2D H+S, 2H H+minor, 2S S+minor)
DONT=(DBL 1 suit, 2C C+higher suit, 2D D+higher suit, 2H H+S, 2S weak S)
HELLO=(DBL penalty, 2C D or Maj/min, 2D long H, 2H H+S, 2S long S)
Landy=(DBL penalty, 2C H+S, 2D long D, 2H long H, 2S long S)
Meckwell=(DBL min's or Maj's, 2C C+Maj, 2D D+Maj, 2H long H, 2S long S)
Wooslwel=(DBL 5Min/4Maj, 2C H+S, 2D H or S, 2H 5H+minor, 2S 5S+minor)

preempts=PRE NAMY
PRE=>S  L  W  (preempts:opening preempts sound rule of 500, light down 2 3 or 4 based on vulnerability, weak only promises length)
NAMY=(preempts:4C and 4D opening bids Namyats)

overT/ODbl= NSF1 NSF2 TOXJS XXNO MAJX2N MINX2N
NSF1=(overT/ODbl:New suit forcing - 1 level - after T.O. Dbl)
NSF2=(overT/OObl:New suit forcing - 2level - after T.O. Dbl)
TOXJS=>F  I  W  (OverT/ODbl:Jump shift ?)
XXNO=(OverT/ODbl:Redouble implies No Fit)
MAJX2N=>LR+  LR  W  NAT (OverT/ODbl:2N after their T.O. dbl of Maj)
MINX2N=>LR+  LR  W  NAT (OverT/ODbl: 2N after their T.O. dbl of min.)

cueBids=MINQ MAJQ ARTQ 
MINQ=>NAT  TO  MICH  TAB (Q/minor)
MAJQ=>NAT  TO  MICH  TAB (Q/Major)
ARTQ=>NAT  TO MICH  TAB (Q-bid/Artif. bid)

Vs.Preempts=VPRETEXT VPRE2N VPREX LEAPMICH
VPRETEXT=4H
VPRE2N=>NAT  LEB (VsPreempts:2N after dbl of weak 2 bid natual,lebensohl)
VPREX=>TO  PEN (VsPreempts:Dbl is takeout,penalty)
LEAPMICH=(Leaping Michaels)

Slam=4N 4C MW 4NO CTL2 DI
4N=>BLKW  RKC  RKC14  KEYC  RKCMM  ModStd(Slam:ace asking standard,RKC1430,RKC MajMin, RKC0314,Keycard, Mode Std)
4C=>NAT  GERB  RKCG (Slam:Gerber ace asking Never used,standard,RKC)
MW=>NAT  0314 1430 (Slam:Minorwood Never used,0314,1430)
4NO=>DOPI  DEPO (Slam:vs. ace asking interference)
CTL2=(Slam:Ctl bids show 1st or 2nd round control)
DI=(Slam:Declarative Interrogative)

suitLeads=SAKX SJT9 SJTX SKJTX SKQT9 SKQX SKT9X SQJX SQT9X ST9X SXX SXXX SXXXX SXXXXX VSUIT SIG
SAKX=1 2 3
SJT9=1 2 3
SJTX=1 2 3
SKJTX=1 2 3 4
SKQT9=1 2 3 4
SKQX=1 2 3
SKT9X=1 2 3 4
SQJX=1 2 3
SQT9X=1 2 3 4
ST9X=1 2 3
SXX=1 2
SXXX=1 2 3
SXXXX=1 2 3 4
SXXXXX=1 2 3 4 5
VSUIT=4  3(length leads vs suit contracts 4th best,3rd5th best)
SIG=A C S(Primary signal to partner's leads attitude,count,suit preference)

NTleads=AKJX AQJX AJT9 AT9X JT9 JTX KJTX KQT9 KQX KT9X QJX QT9X T9X XX XXX XXXX XXXXX VSUIT
AKJX=
AQJX=
AJT9=
AT9X=
JT9=1 2 3
JTX=1 2 3
KJTX=1 2 3 4
KQT9=1 2 3 4
KQX=1 2 3
KT9X=1 2 3 4
QJX=1 2 3
QT9X=1 2 3 4
T9X=1 2 3
XX=1 2
XXX=1 2 3
XXXX=1 2 3 4
XXXXX=1 2 3 4 5
VSUIT=4  3(length leads vs NT contracts 4th best,3rd5th best)

carding=CARD CARDN 1DS 1DN SMITH FORSTER TECHO TSPF SPECIAL(Defensive carding)
CARD=>Standard UDCA(vs suit contracts)
CARDN=>Standard UDCA(vs NT contracts)
1DS=>STD LAV OE UD(First discard vs suit contracts Standard, Lavinthal, Odd/Even, UpsideDown)
1DN=>STD LAV OE UD(First discard vs NT contracts Standard, Lavinthal, Odd/Even, UpsideDown)
SMITH=(Smith Echo)
FORSTER=(Foster Echo)
TECHO=(Trump Echo)
TSPF=(Trump Suit Preference)
SPECIAL=(Special carding)

(In X colon nopts X is one of 2BDFM)
(old convention files users/a/Application\ Data/Bridge\ Captain/Ver\ 6/)
"""

def validate_elem(elem_str):
    """
    Validate bid description element.
    """
    if elem_str[0] in 'CDSH':
        # suit
        if '/' in elem_str:
            # suit code
            suit_codes = elem_str.split('/')
            if len(suit_codes) > 3:
                # multi suit codes, e.g. 'C1/C2/S1' 1st 2nd control, one stopper
                pass
            if len(suit_codes) == 2:
                pass  
            for suit_code in suit_codes[1:]:
                if not suit_code:
                    # there is nothing after / slash?
                    pass
                else:
                    if len(suit_code[0]) > 2:
                        print("Error: suit code has more than 2 chars", elem_str)
                    if suit_code[0] not in 'LBCFKPQSTX':
                        # TODO, decode suit code, also meaning of -K, -Q
                        if suit_code[0] in '-':
                            if suit_code[1] not in 'KQ':
                                print("Error: suit code has invalid char", elem_str)
                        else:
                            print("Error: invalid initial suit code", elem_str)
                    elif suit_code[0] not in 'CSLKQTFPBX':
                        print("this should not happen", elem_str)
    elif elem_str[0] in 'POR':
        # points pattern, rand of bid TODO: what is O stands for?
        if not re.compile('^[-]*[0-9]+[+]*').match(elem_str[1:]):
            if elem_str[-1] in '+':
                elem_str = elem_str[:-1]
            if elem_str[0] in 'PO':
                elem_str = elem_str[1:]
                if elem_str[0] in '-':
                    elem_str = elem_str[1:]
                if not elem_str in 'INV MIN GF CC ZINV MAX GSF GST INV MGF MINV MST NF NT NT1 SF ST WK'.split():
                    print("Error: invalid points pattern2", elem_str)
            elif elem_str != 'RULE15':
                print("Error: invalid points pattern", elem_str)
        # print("R points pattern", elem_str)
        # print("normal points pattern", elem_str)
        # 0-4, 22+, -
    elif elem_str[0] in '!':
        # alert pattern
        if not re.compile('^[1-9][0-9]*').match(elem_str[1:]):
            # TODO: check alerts defined in Alerts.txt
            pass
    elif elem_str[0] in 'XB@^':
        # single char, check BidMeaning.txt
        pass
    elif elem_str[0] in 'L':
        elem_str = elem_str[1:]
        if '.' in elem_str:
            split_dots = elem_str.split('.')
            if len(split_dots) > 3:
                # TODO not sure about the meaning 8.333-9.823
                print("L Error: too many dots", elem_str)
            if len(split_dots) == 3:
                # TODO not sure how to read 2.6-5.6
                if not re.compile('^[0-9].[0-9]+-[0-9]+.[0-9]+').match(elem_str):
                    print("L Error: invalid two dot range", elem_str)
                return
            before_dot = split_dots[0]
            if len(before_dot) != 1:
                if not re.compile('^[0-9][-]+[0-9]+.[0-9]+').match(elem_str):
                    print("L Error: invalid before dot", elem_str)
                return
            if not before_dot in '0123456789':
                print("L Error: invalid level before dot", elem_str)
            elem_str = split_dots[1]
        if not re.compile('^[0-9]-[0-9]').match(elem_str):
            if len(elem_str) < 3:
                if elem_str[-1] in '+':
                    elem_str = elem_str[:-1]
                if elem_str not in '0123456789':
                    print("L Error: invalid level", elem_str)
            elif not re.compile('^[0-9]+[-]+[0-9]+').match(elem_str):
                print("L Error: invalid level range", elem_str)
    elif elem_str[0] in 'YAK':
        if not re.compile('^[<>=]*[0-4][+]*').match(elem_str[1:]):
            print("Y Error: invalid key cards range", elem_str)
    elif elem_str[0] in 'V':
        if elem_str[1] not in 'YNUEF':
            print("V Error: invalid vul", elem_str)
    elif elem_str[0] in 'T':
        if elem_str[1] in 'AKCY':
            if not re.compile('^[<>=]*[0-9][+]*').match(elem_str[2:]):
                print("T Error: invalid range", elem_str)
        else:
            print("T Error: invalid type", elem_str)
    elif elem_str[0] in '{[':
        if elem_str[-1] not in '}]':
            print("Error: invalid bracket", elem_str)
    elif elem_str[0] in 'N':
        if elem_str[1] in 'T':
            if len(elem_str) > 3:
                print("N Error: invalid range", elem_str)
            elif len(elem_str) > 2 and elem_str[2] not in 'NSWY':
                print("N Error: invalid type", elem_str)
        else:
            print("N Error: invalid type", elem_str)
    elif elem_str[0] in 'Q':
        if len(elem_str) < 2 or len(elem_str) > 4:
            print("Q Error: invalid range", elem_str)
        elif len(elem_str) == 2:
            if elem_str[1] not in 'NY':
                print("Q Error: invalid type", elem_str)
        elif len(elem_str) == 3:
            if elem_str[1] not in 'N' or elem_str[2] not in 'T':
                print("Q Error: invalid type3", elem_str)
        else:
            if elem_str[1] not in 'N' or elem_str[2] not in 'S' or elem_str[3] not in '01234':
                print("Q Error: invalid type4", elem_str)
    elif elem_str[0] in 'U':
        if len(elem_str) > 1:
            print("U Error: invalid range", elem_str)
    elif elem_str[0] in '4':
        if len(elem_str) == 2:
            if elem_str[1] not in 'NY':
                print("4 Error: not N or Y", elem_str)
        else:
            print("4 Error: invalid range", elem_str)
    elif elem_str[0] in '*':
        # TODO meaning of DF1,2
        if elem_str[1:] not in 'DF1 DF2 NOBID'.split():
            print("* Error: invalid range", elem_str)
    elif elem_str[0] in '\\':
        if elem_str[1] not in '0235':
            # TODO check convention after number
            print("* Error: invalid range", elem_str)
    else:
        print("Error: invalid bid description element", elem_str)


def validate_desc(desc_str):
    """ validate description line (after =)
    """
    if ' ' in desc_str and ',' in desc_str:
        print("both space and comma", desc_str)
    descriptions = desc_str.split('OR')
    if len(descriptions) > 1:
        for desc in descriptions:
            validate_desc(desc.strip())
            return
    if '"' in desc_str:
        qs = desc_str.split('"')
        if len(qs) > 3:
            print("Error: too many quotes", desc_str)
            return
        desc_str = qs[0]
    if ',' in desc_str:
        desc_str = desc_str.replace(',', ' ')
    elems = desc_str.split()
    for elem in elems:
        validate_elem(elem)

def validate_seq(seq_str):
    """
    Validate seq line (before =).
    """
    for bid in seq_str.split('-'):
        if bid[0] in '@^':
            bid = bid[1:]
        if bid.startswith('('):
            bid = bid.strip('()')
        level = -1
        if bid[0] in '1234567':
            level = int(bid[0])
            suit = bid[1]
            assert suit in 'SHDCN'
        else:
            if bid not in 'XP' and bid not in ['XX']:
                print("Error: invalid bid", bid, seq_str)

def parse_line(line):
    """ verify line is a valid bid convention
    """
    if '=' not in line:
        print("Error: line is not a valid bid convention", line)
        return ''
    seq = line.split('=')
    if len(seq) < 1:
        print("Error: no = ", line)
        return ''
    # if len(seq) > 2:
    #     print('=2+', line)
    validate_seq(seq[0])
    validate_desc('='.join(seq[1:]))
    return seq[0]


def parse_sequence(fname):
    """
    Parse SequnceFile.txt which is in .ini format.
    """
    sequence_dict = {}
    fd = open(fname, 'r')
    while True:
        line = fd.readline()
        if not line:
            break
        line = line.strip()
        if len(line) == 0:
            continue
        # print(line)
        if line.startswith("["):
            key = line.strip("[]")
            sequence_dict[key] = []
        else:
            parse_line(line)
            sequence_dict[key].append(line)
    return sequence_dict


def parse_convention(fname, convt_list):
    """
    Parse {fname}.cnvt which is in .ini format.
    """
    result = []
    fd = open(f'{fname}.cnvt', 'r')
    while True:
        line = fd.readline()
        if not line:
            break
        line = line.strip()
        if len(line) == 0:
            continue
        # print(line)
        k,v = line.split("=")
        if k[0] in 'Z':
            continue
        if k in ['GENS']:
            continue
        if v in ['NAT', 'NUL', 'NONE', 'NATURAL', 'STD']:
            continue
        cnvt = f'{k}>{v}'
        if ':' in cnvt:
            cnvt = cnvt[2:]
        if v[0] not in 'X0123456789_' and cnvt in convt_list:
            result.append(cnvt)
    return result


def check_convt_defined(convt_list):
    """
    Check if all the bidding conventions are defined.
    """
    fnames = 'Goren,Std American,K-S,Precision,Two Over One'
    fname = 'Std American'
    # fnames = fnames.split(',')
    return parse_convention(fname, convt_list)


def fill_hirachy(hirachy, key, seqs):
    """
    Fill the hirachy with the sequence.
    """
    if not hirachy:
        hirachy[key] = {}
        for seq in seqs:
            if ':' in seq:
                seq, desc = seq.split(':')
                hirachy[key][seq] = {desc: {}}
            else:
                hirachy[key][seq] = {}
            
        return True
    to_check = []
    for k in hirachy:
        to_check.append((k, hirachy))

    while to_check:
        k, h = to_check.pop()
        if k == key:
            fill_hirachy(h, key, seqs)
            return True
        for k in h:
            to_check.append((k, h[k]))

    print(f"Error: {key} not found", seqs)
    return False

def check_convt_hirachy():
    """
    Check convention hirachy.
    """
    hirachy = {}
    for line in CONVENTION_HIERACHY.splitlines()[1:]:
        line = line.strip()
        if len(line) == 0:
            continue
        if line.startswith('#'):
            continue
        if '=' not in line:
            continue
        note_start = line.find('(')
        if note_start >= 0:
            line = line[:note_start].strip()
        if len(line) == 0:
            continue
        seq = line.split('=')
        key = seq[0]
        next_sq = seq[1]
        if next_sq.startswith('>'):
            next_sq = next_sq[1:]
        if next_sq == '0..40':
            continue
        if not fill_hirachy(hirachy, key, next_sq.split()):
            return


def bid_match(bid_seqs, compact_bid_line):
    last_bid = bid_seqs[-1]
    if not compact_bid_line.endswith(last_bid):
        return False
    bid_pattern = '.*'.join(bid_seqs)
    if re.compile(f'^{bid_pattern}').match(compact_bid_line):
        return True
    #print(f"Error: {compact_bid_line} not match {bid_seqs[0]}")
    return False


def explain_bid(bid_string, cnvts_list):
    """
    Explain a bid sequence from convention sequcenceFile.txt
    bid_string: whitespace seperated bid string sequence
    """
    rule_set = []
    bid_seqs = bid_string.split()
    for line in cnvts_list:
        if bid_match(bid_seqs, parse_line(line)):
            rule_set.append(line)
    return rule_set


def get_level_bid(bids):
    """
    Get the latest level bid in a bid sequence.
    """
    idx = len(bids) - 1
    while idx >= 0:
        bid = bids[idx]
        if bid.is_pass() or bid.is_double() or bid.is_redouble():
            idx -= 1
            continue
        return bid
    return None

def bid_seq_gen():
    """
    Generate all possible bid sequences
    7n: 7 possible sequences, ppp, drppp, dpprppp, dppp, ppd*3(rppp,pprppp,ppp)
    7s to 7n: 7n can replace any one of the last ppp to form a new start, 3*7=21
    ppp: o, p?, ppo  # even position is own team response
    dppp: d?, dpo, dpp?
    drppp: dro, drp?, drppo
    dpprppp: dppro, dpprp?, dpprppo
    ppdppp: ppd?, ppdpo, ppdpp?
    ppdrppp: ppdro, ppdrp?, ppdrpp?
    ppdpprppp: ppdppro, ppdpprp?, ppdpprppo

    first: 1 pppp
    2nd: 1c start at any seat, 4 possible sequences
    f(level_bid) = 21*(f(level_bid + 1) + f(level_bid + 2) + ... f(7n)) + 7
    f(7s) = 21*(f(7n)) + 7 = 22*7 = 154
    f(7h)+f(7s)+f(7n) = 22^0*7+22^1*7+22^2*7=(22^3-1)/3 = 3549
    f(1c) = (22^35-1)/3
    there are 4 seats, to start 1c-7n, plus first pppp: 4*(22^35-1)/3 + 1
    """
    seq = []
    for bid in Bid.next_bid([]):
        seq.append([bid])
    level_bid = None
    counter = 0
    seq = seq[-3:]

    while len(seq) > 0:
        bids = seq.pop()
        next_bids = Bid.next_bid(bids, get_level_bid(bids))
        print('nb', [str(b) for b in bids], get_level_bid(bids), [str(b) for b in next_bids])
        if next_bids:
            for bid in next_bids:
                #assert bid.level < 1
                seq.append(bids + [bid])
        else:
            counter += 1
            print(counter, [str(b) for b in bids])

    print(counter)


def opening_check(convts_list):
    """ list all openings and its distribution
    """
    for rank in range(1, 8):
        for suit in 'CDHSN':
            bid = f'{rank}{suit}'
            explain_bid(bid, cnvts_list)

if __name__ == '__main__':
    d = parse_sequence('SequenceFile.txt')
    cnvts = check_convt_defined(d.keys())
    cnvts.append('STD')
    cnvts_list = []
    for cnvt in cnvts:
        for line in d[cnvt]:        
            cnvts_list.append(line)
    opening_check(cnvts_list)
    for rank in range(1, 8):
        for suit in 'SHDCN':
            bid = f'{rank}{suit}'    
    explain_bid('1D 2D', cnvts_list)

    # print(d.keys())
    check_convt_hirachy()
    #bid_seq_gen()