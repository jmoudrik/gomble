#!/usr/bin/env python

import math
import copy
from scipy import special, exp, log
import numpy as np
import logging

from kombilo import kombiloNG, libkombilo as lk
from configobj import ConfigObj

import gomill.common

lgam = special.gammaln

PRIOR_WINS = 5
PRIOR_LOSSES = 5


def binomial(n, k, p):
    return exp(lgam(n+1) - lgam(n-k+1) - lgam(k+1) + k*log(p) + (n-k)*log(1.-p))


def binom_integrate(n, k, fr, to, SAMPLES=200):
    assert fr < to
    lp = np.linspace(fr, to, SAMPLES)
    bp = binomial(n, k, lp)
    return bp.sum() / SAMPLES


def wilson_score_interval(c):
    z = 1.96 # 5%
    #z = 0.67 # 50%
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


def WIJM_score(c):
    si = c.wilson_score_interval()
    sc = int2width(si)
    return max(0, c.winrate - 2 * sc)


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
    #print move, format_raw(rc)

    if rc.B and rc.W:
        logging.warn("A cont for both black and white: '%s'"%move)

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
        return "Cont(%s %s, winrate=%.2f%%, samples=%d, wins=%d)"%(
                                                        self.color,
                                                        self.move,
                                                        100*self.winrate,
                                                        self.samples,
                                                        self.wins)


def int2width((l, u)):
    return u - l

def str_int((l, u)):
    return "(%.3f, %.3f)"%(l, u)

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
            print this.move, nb.move, d_samples, d_winrate

            if d_samples <= 0 and d_winrate <= 0:
                # nb is better or equal, skip this
                good = False
                if verbose:
                    print "%s is dominated by %s"%(this.move, nb.move)
                break
            if d_samples >= 0 and d_winrate >= 0:
                # this is better or equal, skip nb
                if verbose:
                    print "%s dominates %s"%(this.move, nb.move)
                nxt.remove(nb)

        if good:
            ret.append(this)

    return ret


def cont_from_pat():
    with open('/home/jm/.kombilo/08/kombilo.cfg', 'r') as fin:
        config = ConfigObj(infile=fin)

    ng = kombiloNG.KEngine()
    #print config['databases']
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
...O...............
...............X...
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
..O.X....,.....X...
...................
...................
...................""", ptype=lk.FULLBOARD_PATTERN)

    ng.patternSearch(p , so)

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
         ]

    return [ Cont(*t) for t in l ]

def main():
    def int_exp(c):
        return binom_exp(c.samples, c.wins, 0.001, 0.5, 200)
    def wilson_width(c):
        si = c.wilson_score_interval()
        return int2width(si)
    def int_score_neg(c):
        return binom_integrate(c.samples, c.wins, 0.001, 0.5)
    def int_score_pos(c):
        return binom_integrate(c.samples, c.wins, 0.5, 0.999)
    def int_score_whole(c):
        return  int_score_neg(c) - int_score_pos(c)
    def int_score_log(c):
        return  log(int_score_neg(c)) - log(int_score_pos(c))

    def add_prior(c):
        cc = copy.copy(c)
        cc.wins += PRIOR_WINS
        cc.samples += PRIOR_WINS + PRIOR_LOSSES
        return cc

    def score(c):
        cc = add_prior(c)
        #return int_score_neg(cc)
        #return int_score_whole(cc)
        return int_score_log(cc)

    #cs = cont_test()
    cs = cont_from_pat()

    if False:
        print len(cs)
        cs = find_nondominated(cs, True)
        print len(cs)

    print
    s = sorted(cs, key=score)
    #s = [ s[0], s[3]]
    for c in s:
        #print
        print repr(c)
        #sc = int2width(wilson_score_interval(c))
        #print "Wilson interval: %s, width=%s"%(str_int(si), sc)
        #print "< 0.5  ", int_score_neg(c)
        #print "< 0.5 P", int_score_neg(add_prior(c))
        #print "> 0.5  ", int_score_pos(c)
        #print "> 0.5 P", int_score_pos(add_prior(c))
        #print "tot   P", int_score_whole(add_prior(c))
        print "log   P", int_score_log(add_prior(c))
        print "log    ", int_score_log(c)
        #print "goodnes", int_score_pos(c)
        #print "exp    ", int_exp(c)
        #print "badness/exp", score(c) / int_exp(c)


if __name__ == "__main__":
    main()
    pass
