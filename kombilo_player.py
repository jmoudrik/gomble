#!/usr/bin/env python
import numpy as np
import logging

import gomill
import gomill.common
from gomill import gtp_engine, gtp_states

from gomble import MoveProbBot
from kombilo_book import MoveFinder


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
                         }

        self.name = None

        self.mf = MoveFinder()
        self.probs = None

    def genmove(self, state, player):
        """
        :returns: gomill.Move_generator_result
        """
        result = gtp_states.Move_generator_result()

        self.probs = self.mf.probs_by_the_book(state.board, player)
        if ( not self.probs
            and state.board.is_empty()
            and player.lower().startswith('b')):
            self.probs = [ ('Q17', 0.55), ('Q16', 0.45) ]

        if self.probs:
            result.move = gomill.common.move_from_vertex(self.probs[0][0],
                                                         state.board.side)
        else:
            result.pass_move = True

        return result

    def handle_name(self, args):
        if self.name is None:
            return self.__class__
        return self.name

    def handle_move_probabilities(self, args):
        if not self.probs:
            return ''
        return '\n'.join( "%s %.6f"%(move, prob) for move, prob in self.probs )

    def get_handlers(self):
        return self.handlers


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # player def
    player = KombiloFusekiPlayer()

    # player => engine => RUN
    engine = make_engine(player)
    gomill.gtp_engine.run_interactive_gtp_session(engine)
