#!/usr/bin/env python
from __future__ import print_function

import sys
import copy
import logging
import subprocess
import numpy as np
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
    
class GtpError(Exception):
    pass

def gtp_cut_response(response):
    assert response
    assert response[0] in '=?'
    if response[0] == '?':
        raise GtpError(response.strip())
    else:
        tail =  re.search(r'^=[0-9]*(.*)$', response, flags=re.DOTALL)
        if tail:
            return tail.groups()[0].strip()
        
        return None

class GtpBot(RWI):
    def __init__(self, bot_cmd):
        self.bot_cmd = bot_cmd
        
        self.p = subprocess.Popen(self.bot_cmd,
                                  stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=None)
        
        self.commands = self.interact('list_commands').split('\n')
        
        assert 'reg_genmove' in self.commands \
               or ('genmove' in self.commands and 'undo' in self.commands)
    
    def write_reg_genmove(self, player):
        if 'reg_genmove' in self.commands:
            self.write('reg_genmove %s'%player)
        else:
            self.write('genmove %s'%player)
            
    def reg_genmove_post(self):
        if not 'reg_genmove' in self.commands:
            self.interact('undo')
    
    def close(self):
        self.p.terminate()
        
    def write(self, gtp_cmd):
        #logging.debug("%s: write %s"%(self.bot_cmd[0], repr(gtp_cmd)))
        self.p.stdin.write(gtp_cmd + "\n")
        self.p.stdin.flush()
        
    def read(self):
        response = self.raw_read()
        return gtp_cut_response(response)
        
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
    
                
class GtpEnsemble(RWI):
    def __init__(self, bots):
        self.bots = bots
        
        self.commands = set.intersection(*[set(b.commands) for b in self.bots])
        
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
        
        Spec as in http://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html
        """
        if raw_line and raw_line[-1] !=  '\n':
            logging.warn("missing newline at the end")
        
        logging.debug("GTP command: %s"%repr(raw_line))
        line = re.sub(r'\s+', ' ', raw_line)
        line = re.sub(r'#.*', '', line)
        cmdline = line.strip().split()
        if not cmdline:
            return None
        
        cmdid = ''
        if re.match('\d+', cmdline[0]):
            cmdid = cmdline[0]
            cmdline = cmdline[1:]
            
        cmd, args = cmdline[0].lower(), cmdline[1:]
        
        ret, err = '', None
        if cmd == "name":
            ret = self.name_version()[0]
        elif cmd == "version":
            ret = self.name_version()[1]
        elif cmd == "protocol_version":
            ret = '2'
        elif cmd == "list_commands":
            ret = '\n'.join(self.commands)
        elif cmd == "genmove":
            try:
                player = args[0]
                ret = handle_genmove(player)
            except GtpError as e:
                err = e.message
        else:
            assert cmd in self.commands
            
            try:
                responses = self.interact(raw_line)
                ret = self.handle_rest(cmd, args, responses)
            except GtpError as e:
                error = e.message
                
        return '%s%s %s\n\n' % ('=' if err is None else '?',
                               cmdid,
                               ret if err is None else err,)
    
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
        #logging.debug("genmove responses: %s"%repr(responses))
        
        # this does undo when previous step did genmove
        for b in self.bots:
            b.reg_genmove_post()
            
        self.interact('play %s %s'%(player, ret))
        return self.choose_move(responses)
            
    def choose_move(self, cmd, args, responses):
        return responses[0]
        
    def handle_rest(self, cmd, args, responses):
        logging.debug("handle rest: %s"%repr(responses))
        assert responses
        
        #logging.debug("GtpEnsemble: no policy, choosing first")
        return responses[0]
    
def raw_input_iterator():
    while True:
        try:
            line = raw_input()
            yield line + '\n'
        except EOFError:
            break
        
def gtp_io(group, iterator):
    for line in iterator:
        resp = group.gtp_one_line(line)
        logging.debug("Response: %s"%repr(resp))
        sys.stdout.write(resp)
        sys.stdout.flush()
        
class VotingEnsemble(GtpEnsemble):
    def __init__(self, bots, weights=None, ties='first'):
        super(VotingEnsemble, self).__init__(bots)
        if weights == None:
            weights = np.ones(len(bots))
        self.weights = weights
            
        assert ties in ['random', 'first']
        self.ties = ties
        
    def choose_move(self, responses):
        logging.debug("VotingEnsemble candidates: %s"%repr(responses))
        
        assert len(responses) == len(self.bots)
        votes = {}
        for bot_num, move in enumerate(responses):
            votes[move] = votes.get(move, 0.0) + self.weights[bot_num]
        logging.debug("VotingEnsemble votes: %s"%repr(votes))
        
        m = max(votes.itervalues())
        
        if self.ties == 'random':
            responses = copy.copy(responses)
            random.shuffle(responses)
            
        for move in responses:
            if votes[move] == m:
                return move
            
        assert False

def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG, filename='LOG')
    
    bots = ["gogui-client haf.ms.mff.cuni.cz 10666", 
            "gnugo --mode gtp --level 10", 
            "gnugo --mode gtp --level 8", 
            "gnugo --mode gtp --level 2", 
            "gnugo --mode gtp --level 1"]
    weights = [1.1, 1.0, 0.8, 0.6, 0.4]
    for num, b in enumerate(bots):
        logging.debug("%d %.2f %s"%(num, weights[num], b))
    
    g = VotingEnsemble(map(GtpBot, (cmd.split() for cmd in bots)),
                       weights=weights)
    
    g.bots[0].commands = filter(lambda c: c!='reg_genmove',
                                g.bots[0].commands )
    
    if True:
    #if False:
        gtp_io(g, raw_input_iterator())
    else:
        gtp_io(g, ['list_commands\n',
                   'boardsize 19\n',
                   'genmove B\n',
                   'genmove W\n',
                   ])
    
    g.interact('quit\n')
    g.close()    
    
if __name__ == "__main__":
    main()
    
    