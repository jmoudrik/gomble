#!/usr/bin/env python
from __future__ import print_function

import sys
import copy
import logging
import subprocess
import numpy as np
import os
import re
import random


class RWI(object):

    def read(self):
        raise NotImplemented

    def write(self, gtp_cmd):
        raise NotImplemented

    def interact(self, gtp_cmd):
        self.write(gtp_cmd)
        return self.read()


def short_repr(obj, size_thresh=100):
    rep = repr(obj)
    if len(rep) < size_thresh:
        return rep

    cutoff = (size_thresh - 3) / 2
    return rep[:cutoff] + '...' + rep[-cutoff:]


class GtpError(Exception):
    pass


def gtp_cut_response(response):
    assert response
    assert response[0] in '=?'
    tail = re.search(r'^([=?])[0-9]*(.*)$', response, flags=re.DOTALL)
    if tail:
        g = tail.groups()
        return g[0] == '=', g[1].strip()

    return None


class GtpBot(RWI):
    def __init__(self, bot_cmd, env_up={}):
        if isinstance(bot_cmd, str):
            bot_cmd = bot_cmd.split()
        self.bot_cmd = bot_cmd

        env = os.environ.copy()
        env.update(env_up)

        self.p = subprocess.Popen(self.bot_cmd,
                                  stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=None,
                                  env=env)

        success, response = self.interact('list_commands')
        assert success
        self.commands = response.split('\n')

        assert 'reg_genmove' in self.commands \
               or ('genmove' in self.commands and 'undo' in self.commands)

        success, response = self.interact('name')
        assert success
        self.name = response

    def __repr__(self):
        return "<%s>" % (self.__class__.__name__)

    def reg_genmove_write(self, player):
        if 'reg_genmove' in self.commands:
            self.write('reg_genmove %s' % player)
        else:
            self.write('genmove %s' % player)

    def reg_genmove_post(self):
        if 'reg_genmove' not in self.commands:
            return self.interact('undo')

    def close(self):
        self.p.terminate()

    def write(self, gtp_cmd):
        logging.debug("%s: write %s" % (repr(self), repr(gtp_cmd)))
        self.p.stdin.write(gtp_cmd + "\n")
        self.p.stdin.flush()

    def read(self):
        response = self.raw_read()
        logging.debug("%s: read %s" % (repr(self), short_repr(response)))
        cut = gtp_cut_response(response)
        return cut

    def raw_read(self):
        lines = []
        prev = "######"
        while True:
            line = self.p.stdout.readline()
            lines.append(line)
            if not line:
                break
            if prev[-1] == "\n" and line == "\n":
                break
            prev = line
        return "".join(lines)


class MoveProbBot(GtpBot):
    """The division int the 3 phases follows from basic patterns
    we may use to implement move_probabilities. The further division
    into write and read is needed because we need to write all commands
    before we start to read. (processing might take some time)

    - pre r/w is expected to do the hard (long) work
    - r/w is expected to gather/compute probs and not take too long
    - cleanup is expected not to take too long

    1) no move_probabilities
        pre_rw = genmove
        rw = just returns (move, 1.0)
        cleanup = undo

    2) move_probabilities, available after genmove, e.g. gnugo, cnn
        pre_rw = genmove
        rw = move_probabilities
        cleanup = undo

    3) no move_probabilities, but other stats (from which we may compute probs)
                              available after genmove (pachi)
        pre_rw = genmove
        rw = pachi-move_statistics
        cleanup = undo
    """

    def __init__(self, *args, **kwargs):
        super(MoveProbBot, self).__init__(*args, **kwargs)

    def move_prob_pre_write(self, player):
        pass

    def move_prob_pre_read(self):
        pass

    def move_prob_write(self, player):
        pass

    def move_prob_read(self, player):
        pass

    def move_prob_cleanup(self, player):
        pass


class MoveProbBotDefault(MoveProbBot):

    def __init__(self, *args, **kwargs):
        super(MoveProbBotDefault, self).__init__(*args, **kwargs)

    def move_prob_pre_write(self, player):
        self.reg_genmove_write(player)

    def move_prob_pre_read(self):
        return self.read()

    def move_prob_write(self, player):
        self.write("move_probabilities")

    def move_prob_read(self):
        success, resp = self.read()
        if not success:
            return None
        toks = [line.split() for line in resp.split('\n')]
        return [(move, float(prob)) for move, prob in toks]

    def move_prob_cleanup(self):
        self.reg_genmove_post()


class Pachi(MoveProbBotDefault):

    def __init__(self, *args, **kwargs):
        super(Pachi, self).__init__(*args, **kwargs)

        assert self.name == "Pachi Distributed"
        # hidden feature ;-)
        self.commands.append("reg_genmove")

    def move_prob_write(self, player):
        self.write("pachi-move_statistics")

    def move_prob_read(self):
        success, resp = self.read()
        if not success:
            return None
        toks = [line.split() for line in resp.split('\n')]
        ps = 0
        for move, playouts, winrate in toks:
            ps += int(playouts)
        return [(move, float(playouts) / ps)
                for move, playouts, winrate in toks]


class KombiloFuseki(MoveProbBotDefault):

    def __init__(self, *args, **kwargs):
        super(KombiloFuseki, self).__init__(*args, **kwargs)

        assert self.name.startswith("Kombilo Fuseki Bot")
        # hidden feature ;-)
        self.commands.append("kombilofuseki-weight")
        self.weight = 1.0

    def move_prob_cleanup(self):
        success, response = self.interact("kombilofuseki-weight")
        assert success
        self.weight = float(response)
        logging.debug('weight: %f'%self.weight)
        self.reg_genmove_post()

def first_good_first_bad(responses):
    """responses = [ (True, 'B4'), (False, 'failed to generate move')]"""
    good, bad = None, None
    for is_good, resp in responses:
        if not good and is_good:
            good = resp or ' '
        if not bad and not is_good:
            bad = resp or 'FAILED'

    return good, bad


class GtpEnsemble(RWI):

    def __init__(self, bots):
        self.bots = bots
        self.commands = set.intersection(*[set(b.commands) for b in self.bots])

        self.movenum = 0
        self.boardsize = 19

        # different bots might have different default boardsize setting
        responses = self.interact('boardsize %d'%self.boardsize)
        good, bad = first_good_first_bad(responses)
        if bad:
            self.close()
            assert False

    def __repr__(self):
        return "<%s>" % (self.__class__.__name__)

    def write(self, gtp_cmd):
        for b in self.bots:
            b.write(gtp_cmd)

    def read(self):
        return [b.read() for b in self.bots]

    def close(self):
        for b in self.bots:
            b.close()

    def name_version(self):
        return "simple ensemble", 0.1

    def gtp_one_line(self, raw_line):
        """
        GTP protocol I/O

        Spec as in
        http://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html
        """
        if raw_line and raw_line[-1] != '\n':
            logging.warn("%s: missing newline at the end" % (repr(self)))

        logging.debug("%s: got cmd %s" % (repr(self), repr(raw_line)))
        line = re.sub(r'\s+', ' ', raw_line)
        line = re.sub(r'#.*', '', line)
        cmdline = line.strip().split()
        if not cmdline:
            return False, None

        cmdid = ''
        if re.match('\d+', cmdline[0]):
            cmdid = cmdline[0]
            cmdline = cmdline[1:]

        cmd, args = cmdline[0].lower(), cmdline[1:]

        ret, error = '', None
        if cmd == "name":
            ret = self.name_version()[0]
        elif cmd == "version":
            ret = self.name_version()[1]
        elif cmd == "protocol_version":
            ret = '2'
        elif cmd == "list_commands":
            ret = '\n'.join(self.commands)
        elif cmd == "quit":
            responses = self.interact(raw_line)
            good, bad = first_good_first_bad(responses)
            if bad:
                logging.error("%s: Err while quitting: %s" %(repr(self), repr(bad)))
            self.close()
            return True, '=\n\n'
        elif cmd == "genmove":
            player = args[0]
            try:
                ret = self.handle_genmove(player)

                logging.debug(
                    '%s: move %d: play %s %s' %
                    (repr(self), self.movenum+1, player, ret))
                self.movenum += 1
            except GtpError as e:
                error = e.message
        else:
            if cmd not in self.commands:
                error = "unknown command"

            if cmd == "boardsize":
                try:
                    boardsize = int(args[0])
                    self.boardsize = boardsize
                except (ValueError, IndexError):
                    error = 'boardsize not an integer'
                    self.boardsize = None

            responses = self.interact(raw_line)
            try:
                ret = self.handle_rest(cmd, args, responses)
            except GtpError as e:
                error = e.message

        return False, '%s%s %s\n\n' % ('=' if error is None else '?',
                                       cmdid,
                                       ret if error is None else error,)

    def handle_genmove(self, player):
        # genmove updates state, but different bots might not agree which would
        # result in inconsistent state were alone genmove used
        # so we need to either
        # 1) use reg_genmove + play
        # 2) or, do sequence of genmove, undo, play

        # this writes either the genmove or reg_genmove
        for b in self.bots:
            b.write_reg_genmove(player)

        responses = self.read()
        good, bad = first_good_first_bad(responses)
        if bad:
            raise GtpError(bad)

        #logging.debug("genmove responses: %s"%repr(responses))

        # this does undo when previous step did genmove
        for b in self.bots:
            b.reg_genmove_post()

        ret = self.choose_move(responses)
        self.interact('play %s %s' % (player, ret))
        return ret

    def choose_move(self, cmd, args, responses):
        return responses[0]

    def handle_rest(self, cmd, args, responses):
        good, bad = first_good_first_bad(responses)
        if bad:
            logging.debug("%s: handle rest failed %s" % (repr(self), repr(bad)))
            raise GtpError(bad)

        assert good
        return good


def raw_input_iterator():
    while True:
        try:
            line = raw_input()
            yield line + '\n'
        except EOFError:
            break


def gtp_io(group, iterator):
    for line in iterator:
        quit, resp = group.gtp_one_line(line)
        logging.debug("<GTP IO>: return: %s" % repr(resp))
        if not resp:
            continue
        sys.stdout.write(resp)
        sys.stdout.flush()

        if quit:
            logging.debug("<GTP IO>: exiting")
            return


class WeightedEnsemble(GtpEnsemble):

    def __init__(self, bots, weights=None):
        super(WeightedEnsemble, self).__init__(bots)
        if weights is None:
            weights = np.ones(len(bots))
        self.weights = weights
        assert len(weights) >= len(bots)

    def weight_for_bot(self, index):
        assert 0 <= index < len(self.weights)

        if self.weights[index] == 'get_from_bot':
            return self.bots[index].weight
        return self.weights[index]


class VotingEnsemble(WeightedEnsemble):

    def __init__(self, bots, weights=None, ties='first'):
        super(VotingEnsemble, self).__init__(bots, weights)
        assert ties in ['random', 'first']
        self.ties = ties

    def choose_move(self, responses):
        logging.debug("%s: candidates: %s" % (repr(self), repr(responses)))

        assert len(responses) == len(self.bots)
        votes = {}
        for bot_num, move in enumerate(responses):
            votes[move] = votes.get(move, 0.0) + self.weight_for_bot(bot_num)
        logging.debug("%s: votes: %s" %(repr(self), repr(votes)))

        m = max(votes.itervalues())

        if self.ties == 'random':
            responses = copy.copy(responses)
            random.shuffle(responses)

        for move in responses:
            if votes[move] == m:
                return move

        assert False


class MoveProbabilityEnsemble(WeightedEnsemble):

    def __init__(self, bots, weights=None, ties='random'):
        super(MoveProbabilityEnsemble, self).__init__(bots, weights)
        for b in self.bots:
            assert isinstance(b, MoveProbBot)

        assert ties in ['random', 'first']
        self.ties = ties

    def get_prob_responses(self, player):
        # genmove updates state, but different bots might not agree which would
        # result in inconsistent state were alone genmove used
        # so we need to either
        # 1) use reg_genmove + play
        # 2) or, do sequence of genmove, undo, play

        # this writes either the genmove or reg_genmove

        # phase 1, pre
        for b in self.bots:
            b.move_prob_pre_write(player)

        moves = [b.move_prob_pre_read() for b in self.bots]
        moves = [m for m in moves if m]

        moves_lower = [m.lower() for s, m in moves]
        if 'pass' in moves_lower:
            return 'pass'
        if 'resign' in moves_lower:
            return 'resign'

        # phase 2, move probs
        for b in self.bots:
            b.move_prob_write(player)
        responses = [b.move_prob_read() for b in self.bots]

        # phase 3, cleanup
        # this does undo when previous step did genmove
        for b in self.bots:
            b.move_prob_cleanup()

        return responses

    def handle_genmove(self, player):
        # get the probabilities from the bots
        responses = self.get_prob_responses(player)

        if responses in ['pass', 'resign']:
            return responses

        # choose the move
        ret = self.choose_move(responses)
        self.interact('play %s %s' % (player, ret))
        return ret

    def choose_move(self, responses):
        assert len(responses) == len(self.bots)
        bv = []
        votes = {}
        for bot_num, resp in enumerate(responses):
            bv.append({})
            if resp is None:
                logging.error(
                    "%s: bot %d: failed probabilities" %
                    (repr(self), bot_num))
                continue
            weight = self.weight_for_bot(bot_num)
            for move, prob in resp:
                vote = prob * weight
                bv[-1][move] = vote
                votes[move] = votes.get(move, 0.0) + vote

        vs = sorted(votes.iteritems(), key=(lambda k_v: k_v[1]), reverse=True)
        msg = ["%s:"%repr(self), "move\tsum\t[votes]"]
        for move, vote in vs[:10]:
            line = ["%s\t%.4f:\t" % (move, vote)]
            for d in bv:
                line.append("%.4f" % d.get(move, 0))
            msg.append(' '.join(line))
        logging.debug('\n'.join(msg))

        m = max(votes.itervalues())
        if self.ties == 'random':
            responses = copy.copy(responses)
            random.shuffle(responses)

        for move, vote in votes.iteritems():
            if vote == m:
                return move

        assert False


def main():
    logging.basicConfig(format='__ %(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG, filename='LOG')

    bots = [# MoveProbBotDefault("gogui-client haf.ms.mff.cuni.cz 10666"),
            # MoveProbBotDefault("gnugo --mode gtp --chinese-rules "
            #                    "--capture-all-dead --level 10"),
            KombiloFuseki("./kombilo_player.py"),
            Pachi('./runpachi.sh -t =10000 slave'),
            #Pachi('./runpachi.sh -t =5000 slave'),
            ]
    weights = ['get_from_bot', 0.5, 1.0, 1.0]
    #for num, b in enumerate(bots):
        #logging.debug("bot #%d\nweight: %s\nbot: %s" % (num, weights[num], b))

    #g = VotingEnsemble(map(GtpBot, bots), weights=weights, ties='random')

    g = MoveProbabilityEnsemble(bots, weights=weights)

    if True:
        # if False:
        gtp_io(g, raw_input_iterator())
    else:
        gtp_io(g, ['list_commands\n',
                   'boardsize 19\n',
                   'genmove B\n',
                   'genmove W\n',
                   'quit\n'
                   ])


if __name__ == "__main__":
    os.chdir(os.path.dirname(sys.argv[0]))
    main()
    pass
