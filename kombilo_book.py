#!/usr/bin/env python

import math
import copy
from scipy import special, exp, log
import numpy as np
import logging
import collections


from kombilo import kombiloNG, libkombilo as lk
from configobj import ConfigObj

import gomill.common
import gomill.sgf
import gomill.sgf_moves

lgam = special.gammaln

PRIOR_WINS = 5
PRIOR_LOSSES = 5
INTEGRATE_SAMPLES = 1000
INTEGRATE_LINSPACE = np.linspace(0.0001, 0.9999, 2*INTEGRATE_SAMPLES)
INTEGRATE_LINSPACE_LOW = INTEGRATE_LINSPACE <= 0.5
INTEGRATE_LINSPACE_HIGH = np.logical_not(INTEGRATE_LINSPACE_LOW)

MoveFinderRet = collections.namedtuple('MoveFinderRet', 'move probs scores weight')
MoveValue = collections.namedtuple('MoveValue', 'move value')


def binomial(n, k, p):
    return exp(lgam(n+1) - lgam(n-k+1) - lgam(k+1) + k*log(p) + (n-k)*log(1.-p))


def binom_integrate(n, k, fr, to):
    assert fr < to
    lp = np.linspace(fr, to, INTEGRATE_SAMPLES)
    bp = binomial(n, k, lp)
    return bp.sum() / INTEGRATE_SAMPLES


def log_diff_binom_integrate(n, k):
    # faster (and a bit more precise)
    # alternative
    bp = binomial(n, k, INTEGRATE_LINSPACE)
    lo = INTEGRATE_LINSPACE_LOW * bp
    hi = INTEGRATE_LINSPACE_HIGH * bp

    return log(lo.sum()) - log(hi.sum())


def score_int_neg(c):
    return binom_integrate(c.samples, c.wins, 0.0001, 0.5)


def score_int_pos(c):
    return binom_integrate(c.samples, c.wins, 0.5, 0.9999)


def score_diff_log_int(c):
    return log_diff_binom_integrate(c.samples, c.wins)

def add_prior(c):
    cc = copy.copy(c)
    cc.wins += PRIOR_WINS
    cc.samples += PRIOR_WINS + PRIOR_LOSSES
    return cc

def log_score_with_prior(c):
    cc = add_prior(c)
    return score_diff_log_int(cc)

def log_scores2probs(scores):
    scores = scores - scores.min() + 1
    scores_over = 1 / scores
    return scores_over / scores_over.sum()

def freq_scores2probs(scores):
    scores = np.abs(scores)
    return scores / scores.sum()

def wilson_score_interval(c):
    z = 1.96  # 5%
    # z = 0.67 # 50%
    p = c.winrate
    n = float(c.samples)
    PM = z * math.sqrt(p*(1-p) / n + z*z / (4 * n*n))
    A = 1 / (1 + z*z/n)
    B = p + z*z/(2*n)

    return (A * (B - PM), A * (B + PM))


def agresti_coul(c):
    z = 1.96
    n = float(c.samples) + z*z
    p = (c.wins + z*z / 2)/n
    PM = z * math.sqrt(p*(1-p) / n)

    return (p - PM, p + PM)


def format_raw(rc):
    props = ['B', 'W', 'wB', 'wW', 'lB', 'lW', 'tB', 'tW']
    return ', '.join("%s=%d" % (prop, getattr(rc, prop)) for prop in props)


def cont_from_raw(rc):
    if rc.B:
        color, samples, wins, losses = 'black', rc.B, rc.wB, rc.lB
    else:
        # kombilo/lk/pattern.h see ~ line 242 for struct of rc
        color, samples, wins, losses = 'white', rc.W, rc.lW, rc.wW

    move = gomill.common.format_vertex((18 - rc.y, rc.x))
    c = Cont(color, move, samples, wins, losses)
    # print move, format_raw(rc)

    if rc.B and rc.W:
        logging.warn("A cont for both black and white: '%s'" % move)

    return c


class Cont:

    def __init__(self, color, move, samples, wins, losses='count'):
        self.color = color
        self.move = move
        self.samples = samples
        self.wins = wins
        self.losses = losses
        if losses == 'count':
            self.losses = samples - wins
        self.winrate = self.wins / float(self.samples)

    def __repr__(self):
        return "Cont(%s %s, winrate=%.2f%%, samples=%d, wins=%d)" % (
                                                        self.color,
                                                        self.move,
                                                        100*self.winrate,
                                                        self.samples,
                                                        self.wins)


def find_nondominated(cs, verbose=False):
    nxt = set(cs)
    ret = []

    while nxt:
        this = nxt.pop()
        if verbose:
            print repr(this)
        good = True
        for nb in copy.copy(nxt):
            d_samples = this.samples - nb.samples
            d_winrate = this.winrate - nb.winrate

            if d_samples <= 0 and d_winrate <= 0:
                # nb is better or equal, skip this
                good = False
                if verbose:
                    print "%s is dominated by %s" % (this.move, nb.move)
                break
            if d_samples >= 0 and d_winrate >= 0:
                # this is better or equal, skip nb
                if verbose:
                    print "%s dominates %s" % (this.move, nb.move)
                nxt.remove(nb)
        if good:
            ret.append(this)

    return ret


def format_board(board):
    A = [['.'] * board.side for _ in xrange(board.side)]

    for (color, (row, col)) in board.list_occupied_points():
        # gomill has origin in lower left corner
        row = board.side - row - 1
        A[row][col] = 'X' if color.lower().startswith('b') else 'O'

    return '\n'.join(''.join(l) for l in A)


class MoveFinder:

    def __init__(self, score_type='freq', config_file='/home/jm/.kombilo/08/kombilo.cfg'):
        self.config_file = config_file
        with open(config_file, 'r') as fin:
            config = ConfigObj(infile=fin)

        assert score_type in ['freq', 'log']
        self.score_type = score_type

        self.ng = kombiloNG.KEngine()
        # print config['databases']
        self.ng.gamelist.populateDBlist(config['databases'])
        self.ng.loadDBs()

        # default search options
        self.so = lk.SearchOptions()
        self.so.fixedColor = 1
        self.so.moveLimit = 10000
        self.so.searchInVariations = False

    def find_continuations(self, board, color):
        #assert board.side == 19
        color = 'black' if color.lower().startswith('b') else 'white'
        logging.debug(color)
        self.so.nextMove = 1 if color == 'black' else 2

        pattern = format_board(board)
        logging.debug('pattern:\n' + pattern)
        self.ng.patternSearch(kombiloNG.Pattern(pattern, ptype=lk.FULLBOARD_PATTERN),
                              self.so,
                              update_gamelist=False)

        return map(cont_from_raw, self.ng.continuations)

    def continuation_score(self, c):
        """badness of a continuation, the less, the better"""
        if self.score_type == 'freq':
            return - c.samples
        if self.score_type == 'log':
            return log_score_with_prior(c)
        assert False

    def scores2probs(self, scores):
        if self.score_type == 'freq':
            return freq_scores2probs(scores)
        if self.score_type == 'log':
            return log_scores2probs(scores)
        assert False

    def conts2weight(self, cs):
        THRESHOLD=0.48
        if not cs:
            return None
        bwr = cs[0].winrate
        return min(bwr / THRESHOLD, 1.0)

    def by_the_book(self, board, color, verbose=False):
        """From position given by ``pattern``, returns tuple:
        * the best continuation
        * probabilities of next moves, proportional to their score
        from the database for a player ``color``.

        If the best move is not good enough, returns (None, [probs])
        """
        # XXX
        # FIXME kombiloNG breaks on empty board for some reason
        #   it returns nothing and breaks consequent searches as well
        #   (to return nothing as well)
        #
        if (board.side == 19
            and board.is_empty()
            and color.lower().startswith('b')):
            return MoveFinderRet('Q17',
                                 [ MoveValue('Q17', 0.55), MoveValue('Q16', 0.45) ],
                                 [ MoveValue('Q17', 0.55), MoveValue('Q16', 0.45) ],
                                 1.0)

        cs = self.find_continuations(board, color)
        move = self.conts2move(cs, verbose)
        cs, probs, scores = self.conts2all(cs)

        logging.debug("Continuations: " + str(cs))
        logging.debug("Probs: " + str(probs))
        logging.debug("Scores: " + str(scores))

        ret_probs = None
        if probs is not None:
            ret_probs = [MoveValue(cont.move, prob)
                                for cont, prob in zip(cs, probs)]
        ret_scores = None
        if scores is not None:
            ret_scores = [MoveValue(cont.move, score)
                                for cont, score in zip(cs, scores)]
        return MoveFinderRet(move, ret_probs, ret_scores, self.conts2weight(cs))

    def conts2all(self, cs):
        if not cs:
            return None, None, None
        scores = np.array([self.continuation_score(c) for c in cs])
        scores, cs = zip(*sorted(zip(scores, cs)))
        scores = np.array(scores, dtype='float')
        return cs, self.scores2probs(scores), scores

    def conts2move(self, cs, verbose=False):
        if not cs:
            return None

        #cs = find_nondominated(cs, False)
        cs.sort(key=self.continuation_score)
        best_move = cs[0]

        if verbose:
            for c in cs:
                print repr(c)
                #print "< 0.5 P", score_int_neg(add_prior(c))
                #print "> 0.5 P", score_int_pos(add_prior(c))
                print "score", self.continuation_score(c)

        if best_move.winrate < 0.48:
            return None

        return best_move.move


def cont_from_pat():
    with open('/home/jm/.kombilo/08/kombilo.cfg', 'r') as fin:
        config = ConfigObj(infile=fin)

    ng = kombiloNG.KEngine()
    # print config['databases']
    ng.gamelist.populateDBlist(config['databases'])
    ng.loadDBs()

    so = lk.SearchOptions()
    so.fixedColor = 1
    so.nextMove = 0  # both
    so.nextMove = 1  # black
    so.nextMove = 2  # white
    so.moveLimit = 10000  # black
    so.searchInVariations = False

    p = kombiloNG.Pattern("""
...................
...................
...............X...
...................
...................
...................
...................
...................
...................
...,.....,.....,...
...................
...................
...................
...................
...................
.........,.........
...................
...................
...................""", ptype=lk.FULLBOARD_PATTERN)

    ng.patternSearch(p, so)

    return map(cont_from_raw, ng.continuations)


def cont_test():
    # ( color, move, samples, wins )

    l = [
          ('black', 'nahodny maly', 1, 1),
          ('black', 'nahodny', 2, 2),
          ('black', 'nahodny vetsi', 3, 3),
          ('black', 'pekny', 100, 60),
          ('black', 'super', 50, 40),
          ('black', 'osklivy', 100, 40),
          ('black', 'hruza', 50, 10),
         ]

    l = [
          ('black', 'nahodny', 2, 2),
          ('black', 'nahodny vetsi', 3, 3),
          ('black', 'nadprumerny', 26, 15),
          ('black', 'nejlepsi', 100, 60),
          ('black', 'prumer', 10000, 5000),
         ]

    return [Cont(*t) for t in l]


def test_speed():
    c = Cont('black', 'prumer', 10000, 5000)

    for a in xrange(1000):
        #s = score_int_neg(c)
        #s = score_int_pos(c)
        s = score_diff_log_int(c)


def main():
    def score(c):
        cc = add_prior(c)
        # return score_int_neg(c)
        # return score_int_pos(c)
        return score_diff_log_int(cc)

    #cs = cont_test()
    cs = cont_from_pat()

    # if True:
    if False:
        print len(cs)
        cs = find_nondominated(cs, False)
        print len(cs)

    s = sorted(cs, key=score)
    for c in s:
        print repr(c)
        print "< 0.5 P", score_int_neg(add_prior(c))
        print "> 0.5 P", score_int_pos(add_prior(c))
        print "diff  P", score_diff_log_int(add_prior(c))


def get_board(move=5):
    with open("test2.sgf", 'r') as fin:
        game = gomill.sgf.Sgf_game.from_string(fin.read())

    board, movepairs = gomill.sgf_moves.get_setup_and_moves(game)
    for color, move in movepairs[:move]:
        if move:
            row, col = move
            board.play(row, col, color)

    return board

def test_search():
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    mf = MoveFinder()
    print "init done"

    #print format_board(board)
    #print mf.move_by_the_book(board, 'w', True)
    board = get_board(0)
    mf.probs_by_the_book(board, 'B')

    board = get_board(1)
    mf.probs_by_the_book(board, 'W')

if __name__ == "__main__":
    # main()
    #test_speed()
    test_search()
    pass
