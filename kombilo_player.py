#!/usr/bin/env python
import numpy as np
import logging

import gomill
import gomill.common
from gomill import gtp_engine, gtp_states

from gomble import MoveProbBot
import kombilo_book
from kombilo_book import MoveFinder, MoveFinderRet, MoveValue


def make_engine(player):
    """Return a Gtp_engine_protocol which runs the specified player."""
    gtp_state = gtp_states.Gtp_state(move_generator=player.genmove)

    engine = gtp_engine.Gtp_engine_protocol()
    engine.add_protocol_commands()
    engine.add_commands(gtp_state.get_handlers())
    engine.add_commands(player.get_handlers())
    return engine

class KombiloFusekiPlayer(object):

    def __init__(self):
        self.handlers = {'name': self.handle_name,
                         'move_probabilities': self.handle_move_probabilities,
                         'kombilofuseki-weight': self.handle_weight,
                         }
        self.name = "Kombilo Fuseki Bot, v0.1"
        self.mf = MoveFinder('freq')
        self.mps = MoveFinderRet(None, None, None, None)

    def genmove(self, state, player):
        """
        :returns: gomill.Move_generator_result
        """
        logging.debug("KombiloFusekiPlayer.genmove()")
        result = gtp_states.Move_generator_result()

        self.mps = self.mf.by_the_book(state.board, player)
        if self.mps.move:
            result.move = gomill.common.move_from_vertex(self.mps.move,
                                                         state.board.side)
        else:
            result.pass_move = True

        return result

    def handle_name(self, args=[]):
        if self.name is None:
            return self.__class__
        return self.name

    def handle_move_probabilities(self, args):
        logging.debug("KombiloFusekiPlayer.handle_move_probabilities()")
        if not self.mps.probs:
            return ''
        return '\n'.join( "%s %.6f"%(move, prob) for move, prob in self.mps.probs )

    def handle_weight(self, args):
        logging.debug("KombiloFusekiPlayer.handle_weight()")
        if self.mps.weight is None:
            return ''
        return '%.3f'%(self.mps.weight)

    def get_handlers(self):
        return self.handlers


if __name__ == "__main__":
    logging.basicConfig(format='KP %(asctime)s %(levelname)s: %(message)s',
                        #level=logging.INFO)
                        level=logging.DEBUG)
    # player def
    player = KombiloFusekiPlayer()

    # player => engine => RUN
    engine = make_engine(player)
    gomill.gtp_engine.run_interactive_gtp_session(engine)
