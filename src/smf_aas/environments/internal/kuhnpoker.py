"""
Kuhn Poker Environment.

Kuhn Poker is a simplified poker game used extensively in game theory
research. It uses a 3-card deck (Jack, Queen, King) and two players.

Game Rules
----------
1. Each player antes 1 chip
2. Each player receives one card
3. Player 0 acts first: pass or bet (1 chip)
4. Player 1 responds to the action:
   - If player 0 passed: pass (showdown) or bet (1 chip)
   - If player 0 bet: fold (player 0 wins) or call (1 chip, showdown)
5. If player 1 bet after player 0 passed:
   - Player 0 can fold (player 1 wins) or call (1 chip, showdown)
6. At showdown, higher card wins the pot

References
----------
.. [1] Kuhn, H. W. (1950). "A simplified two-person poker".
       Contributions to the Theory of Games, 1, 97-103.

Game Properties
---------------
- Players: 2
- Information: Imperfect (hidden cards)
- Determinism: Stochastic (card dealing)
- Actions: 0 = pass/fold, 1 = bet/call
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from ..base import GameEnvironment, GameState


class KuhnPokerEnv(GameEnvironment):
    """Kuhn Poker environment.
    
    State observation (6 elements):
        [0:3] - One-hot encoding of current player's card (J=0, Q=1, K=2)
        [3:6] - Action history encoding
    
    Actions:
        0 = pass (if first action) or fold (after opponent bet)
        1 = bet (if first action) or call (after opponent bet)
    
    Parameters
    ----------
    seed : int, optional
        Random seed for card dealing.
    
    Examples
    --------
    >>> env = KuhnPokerEnv(seed=42)
    >>> state = env.reset()
    >>> state.legal_actions
    [0, 1]
    >>> state, done = env.step(1)  # Bet
    """
    
    # Card values: Jack=0, Queen=1, King=2
    _JACK, _QUEEN, _KING = 0, 1, 2
    
    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._cards: List[int] = []
        self._history: List[int] = []
        self._current: int = 0
        self._done: bool = False
    
    @property
    def name(self) -> str:
        return "KuhnPoker"
    
    @property
    def num_actions(self) -> int:
        return 2  # pass/fold or bet/call
    
    @property
    def num_players(self) -> int:
        return 2
    
    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (6,)  # 3 for card + 3 for history
    
    def reset(self) -> GameState:
        """Deal cards and start new hand."""
        # Shuffle and deal
        deck = [self._JACK, self._QUEEN, self._KING]
        self._rng.shuffle(deck)
        self._cards = [deck[0], deck[1]]
        
        self._history = []
        self._current = 0
        self._done = False
        
        return self._make_state()
    
    def step(self, action: int) -> Tuple[GameState, bool]:
        """Execute action."""
        if self._done:
            raise ValueError("Game is already finished")
        
        if action not in [0, 1]:
            raise ValueError(f"Invalid action {action}: must be 0 or 1")
        
        self._history.append(action)
        
        # Check if game ends
        terminal, returns = self._check_terminal()
        
        if terminal:
            self._done = True
            return self._make_state(is_terminal=True, returns=returns), True
        
        # Switch player
        self._current = 1 - self._current
        return self._make_state(), False
    
    def _check_terminal(self) -> Tuple[bool, Optional[List[float]]]:
        """Check if game is terminal and compute returns."""
        h = self._history
        
        if len(h) == 2:
            if h == [0, 0]:
                # pass-pass: showdown for ante (pot=2)
                return True, self._showdown(pot=2)
            elif h == [0, 1]:
                # pass-bet: player 0 must respond
                return False, None
            elif h == [1, 0]:
                # bet-fold: player 0 wins ante (pot=2)
                return True, [1.0, -1.0]
            elif h == [1, 1]:
                # bet-call: showdown (pot=4)
                return True, self._showdown(pot=4)
        
        elif len(h) == 3:
            if h[2] == 0:
                # pass-bet-fold: player 1 wins ante (pot=2)
                return True, [-1.0, 1.0]
            else:
                # pass-bet-call: showdown (pot=4)
                return True, self._showdown(pot=4)
        
        return False, None
    
    def _showdown(self, pot: int) -> List[float]:
        """Determine winner at showdown."""
        # Higher card wins
        if self._cards[0] > self._cards[1]:
            # Player 0 wins: gets pot/2 (net profit)
            return [pot / 2, -pot / 2]
        else:
            # Player 1 wins
            return [-pot / 2, pot / 2]
    
    def _make_state(
        self,
        is_terminal: bool = False,
        returns: Optional[List[float]] = None
    ) -> GameState:
        """Create observation for current player."""
        obs = np.zeros(6, dtype=np.float32)
        
        # One-hot encode current player's card
        obs[self._cards[self._current]] = 1.0
        
        # Encode action history (1-indexed to distinguish from no action)
        for i, a in enumerate(self._history[:3]):
            obs[3 + i] = float(a + 1)
        
        return GameState(
            observation=obs,
            legal_actions=[] if is_terminal else [0, 1],
            current_player=self._current,
            is_terminal=is_terminal,
            returns=returns
        )
    
    def get_opponents(self, seed: int = 42) -> Dict[str, Any]:
        """Get opponent policies."""
        return {
            'random': _RandomPolicy(seed),
            'always_call': _AlwaysCallPolicy(),
            'simple': _SimpleStrategy(seed + 1),
        }


class _RandomPolicy:
    """Uniform random policy."""
    
    def __init__(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        return int(self._rng.choice(legal_actions))


class _AlwaysCallPolicy:
    """Always bet/call."""
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        return 1 if 1 in legal_actions else 0


class _SimpleStrategy:
    """Simple rule-based strategy.
    
    - King: always bet
    - Jack: always pass/fold
    - Queen: randomize
    """
    
    def __init__(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        # Decode card from observation
        card = int(np.argmax(observation[:3]))
        
        if card == 2:  # King
            return 1
        elif card == 0:  # Jack
            return 0
        else:  # Queen
            return int(self._rng.choice([0, 1]))
