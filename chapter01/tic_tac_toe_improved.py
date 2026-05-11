#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Jan Hakenberg(jan.hakenberg@gmail.com)                         #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

"""
Improved tic-tac-toe with reinforcement learning.

This module implements a tic-tac-toe game where AI players learn optimal
strategies using temporal difference learning.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pickle

# ==================== Configuration ====================
BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS

# Training hyperparameters
DEFAULT_STEP_SIZE = 0.1
DEFAULT_EPSILON = 0.1
TRAINING_EPSILON = 0.01
EVAL_EPSILON = 0.0

# Value estimations
WIN_VALUE = 1.0
LOSE_VALUE = 0.0
TIE_VALUE = 0.5
INITIAL_VALUE = 0.5

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class State:
    """Represents a tic-tac-toe board state."""

    def __init__(self) -> None:
        """Initialize an empty board state.
        
        The board is represented by an n * n array:
        - 1: first player's piece
        - -1: second player's piece
        - 0: empty position
        """
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
        self.winner: Optional[int] = None
        self.hash_val: Optional[int] = None
        self.end: Optional[bool] = None

    def hash(self) -> int:
        """Compute unique hash value for this state.
        
        Returns:
            int: Hash value representing this board configuration.
        """
        if self.hash_val is None:
            self.hash_val = 0
            for i in np.nditer(self.data):
                self.hash_val = self.hash_val * 3 + int(i) + 1
        return self.hash_val

    def is_end(self) -> bool:
        """Check if the game has ended.
        
        Returns:
            bool: True if game is over (win, lose, or tie), False otherwise.
        """
        if self.end is not None:
            return self.end

        # Check rows, columns, and diagonals
        results = []
        
        # Check rows
        for i in range(BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))
        
        # Check columns
        for i in range(BOARD_COLS):
            results.append(np.sum(self.data[:, i]))

        # Check diagonals
        trace = np.sum([self.data[i, i] for i in range(BOARD_ROWS)])
        reverse_trace = np.sum([self.data[i, BOARD_ROWS - 1 - i] for i in range(BOARD_ROWS)])
        results.append(trace)
        results.append(reverse_trace)

        # Check for winner
        for result in results:
            if result == BOARD_ROWS:  # All player 1
                self.winner = 1
                self.end = True
                return self.end
            if result == -BOARD_ROWS:  # All player 2
                self.winner = -1
                self.end = True
                return self.end

        # Check for tie
        if np.sum(np.abs(self.data)) == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end

        # Game is still ongoing
        self.end = False
        return self.end

    def next_state(self, i: int, j: int, symbol: int) -> 'State':
        """Create a new state with a move applied.
        
        Args:
            i: Row index.
            j: Column index.
            symbol: 1 or -1, representing the player.
            
        Returns:
            State: New state object with the move applied.
        """
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

    def print_state(self) -> None:
        """Print the board in a human-readable format."""
        for i in range(BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(BOARD_COLS):
                if self.data[i, j] == 1:
                    token = '*'
                elif self.data[i, j] == -1:
                    token = 'x'
                else:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-------------')


def get_all_states_impl(
    current_state: State,
    current_symbol: int,
    all_states: Dict[int, Tuple[State, bool]]
) -> None:
    """Recursively generate all possible board states.
    
    Args:
        current_state: Current board state.
        current_symbol: Current player's symbol (1 or -1).
        all_states: Dictionary to store all states (modified in place).
    """
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if current_state.data[i, j] == 0:
                new_state = current_state.next_state(i, j, current_symbol)
                new_hash = new_state.hash()
                if new_hash not in all_states:
                    is_end = new_state.is_end()
                    all_states[new_hash] = (new_state, is_end)
                    if not is_end:
                        get_all_states_impl(new_state, -current_symbol, all_states)


def get_all_states() -> Dict[int, Tuple[State, bool]]:
    """Generate all possible board states.
    
    Returns:
        dict: Mapping of state hash to (State, is_end) tuples.
    """
    current_symbol = 1
    current_state = State()
    all_states: Dict[int, Tuple[State, bool]] = {}
    all_states[current_state.hash()] = (current_state, current_state.is_end())
    get_all_states_impl(current_state, current_symbol, all_states)
    return all_states


# Pre-compute all possible board configurations
ALL_STATES = get_all_states()


class Judger:
    """Manages game play between two players."""

    def __init__(self, player1: 'BasePlayer', player2: 'BasePlayer') -> None:
        """Initialize the judger with two players.
        
        Args:
            player1: First player (moves first, symbol = 1).
            player2: Second player (symbol = -1).
        """
        self.p1 = player1
        self.p2 = player2
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol)
        self.p2.set_symbol(self.p2_symbol)
        self.current_state = State()

    def reset(self) -> None:
        """Reset both players."""
        self.p1.reset()
        self.p2.reset()

    def alternate(self):
        """Alternate between players indefinitely."""
        while True:
            yield self.p1
            yield self.p2

    def play(self, print_state: bool = False) -> int:
        """Play one complete game.
        
        Args:
            print_state: If True, print board after each move.
            
        Returns:
            int: 1 if player1 won, -1 if player2 won, 0 if tie.
        """
        alternator = self.alternate()
        self.reset()
        current_state = State()
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        if print_state:
            current_state.print_state()
        
        while True:
            player = next(alternator)
            i, j, symbol = player.act()
            next_state_hash = current_state.next_state(i, j, symbol).hash()
            current_state, is_end = ALL_STATES[next_state_hash]
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)
            if print_state:
                current_state.print_state()
            if is_end:
                return current_state.winner


class BasePlayer:
    """Base class for players."""

    def __init__(self) -> None:
        """Initialize player."""
        self.symbol = 0

    def reset(self) -> None:
        """Reset player state for a new game."""
        raise NotImplementedError

    def set_state(self, state: State) -> None:
        """Update player's current state."""
        raise NotImplementedError

    def set_symbol(self, symbol: int) -> None:
        """Set player's symbol (1 or -1)."""
        self.symbol = symbol

    def act(self) -> Tuple[int, int, int]:
        """Choose an action and return (row, col, symbol)."""
        raise NotImplementedError


class Player(BasePlayer):
    """AI player using temporal difference learning."""

    def __init__(
        self,
        step_size: float = DEFAULT_STEP_SIZE,
        epsilon: float = DEFAULT_EPSILON
    ) -> None:
        """Initialize AI player.
        
        Args:
            step_size: Learning rate for value updates.
            epsilon: Exploration probability (0 = greedy, 1 = random).
        """
        super().__init__()
        self.estimations: Dict[int, float] = {}
        self.step_size = step_size
        self.epsilon = epsilon
        self.states: List[State] = []
        self.greedy: List[bool] = []

    def reset(self) -> None:
        """Reset for a new game."""
        self.states = []
        self.greedy = []

    def set_state(self, state: State) -> None:
        """Record state visit for learning."""
        self.states.append(state)
        self.greedy.append(True)

    def set_symbol(self, symbol: int) -> None:
        """Set symbol and initialize value estimations.
        
        Args:
            symbol: 1 or -1, representing the player.
        """
        self.symbol = symbol
        self.estimations = {}
        
        for hash_val, (state, is_end) in ALL_STATES.items():
            if is_end:
                if state.winner == symbol:
                    self.estimations[hash_val] = WIN_VALUE
                elif state.winner == 0:
                    self.estimations[hash_val] = TIE_VALUE
                else:
                    self.estimations[hash_val] = LOSE_VALUE
            else:
                self.estimations[hash_val] = INITIAL_VALUE

    def backup(self) -> None:
        """Update value estimations using temporal difference learning."""
        states = [state.hash() for state in self.states]

        for i in reversed(range(len(states) - 1)):
            state_hash = states[i]
            td_error = self.greedy[i] * (
                self.estimations[states[i + 1]] - self.estimations[state_hash]
            )
            self.estimations[state_hash] += self.step_size * td_error

    def act(self) -> Tuple[int, int, int]:
        """Choose action: explore randomly or exploit greedily.
        
        Returns:
            tuple: (row, col, symbol)
        """
        state = self.states[-1]
        next_positions = []
        next_states = []
        
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    next_positions.append([i, j])
                    next_states.append(state.next_state(i, j, self.symbol).hash())

        # Exploration
        if np.random.rand() < self.epsilon:
            action = next_positions[np.random.randint(len(next_positions))]
            self.greedy[-1] = False
            return int(action[0]), int(action[1]), self.symbol

        # Exploitation: choose best value, random tie-break
        values = [
            (self.estimations[hash_val], pos)
            for hash_val, pos in zip(next_states, next_positions)
        ]
        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)
        best_action = values[0][1]
        
        return int(best_action[0]), int(best_action[1]), self.symbol

    def save_policy(self, filepath: Optional[str] = None) -> None:
        """Save learned policy to file.
        
        Args:
            filepath: Path to save file. Defaults to 'policy_<first/second>.bin'.
        """
        if filepath is None:
            filepath = f"policy_{'first' if self.symbol == 1 else 'second'}.bin"
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.estimations, f)
            logger.info(f"Policy saved to {filepath}")
        except IOError as e:
            logger.error(f"Failed to save policy to {filepath}: {e}")

    def load_policy(self, filepath: Optional[str] = None) -> None:
        """Load learned policy from file.
        
        Args:
            filepath: Path to load file. Defaults to 'policy_<first/second>.bin'.
        """
        if filepath is None:
            filepath = f"policy_{'first' if self.symbol == 1 else 'second'}.bin"
        
        try:
            with open(filepath, 'rb') as f:
                self.estimations = pickle.load(f)
            logger.info(f"Policy loaded from {filepath}")
        except FileNotFoundError:
            logger.warning(f"Policy file {filepath} not found. Using default estimations.")
        except IOError as e:
            logger.error(f"Failed to load policy from {filepath}: {e}")


class HumanPlayer(BasePlayer):
    """Human player interface.
    
    Input mapping:
    | q | w | e |
    | a | s | d |
    | z | x | c |
    """

    def __init__(self) -> None:
        """Initialize human player."""
        super().__init__()
        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.state: Optional[State] = None

    def reset(self) -> None:
        """Reset for new game (no-op for human)."""
        pass

    def set_state(self, state: State) -> None:
        """Update current state."""
        self.state = state

    def act(self) -> Tuple[int, int, int]:
        """Get human input and return action.
        
        Returns:
            tuple: (row, col, symbol)
            
        Raises:
            ValueError: If input is invalid.
        """
        if self.state is None:
            raise RuntimeError("State not set")
        
        self.state.print_state()
        
        while True:
            try:
                key = input("Input your position (q/w/e/a/s/d/z/x/c): ").lower()
                if key not in self.keys:
                    print(f"Invalid key. Choose from: {', '.join(self.keys)}")
                    continue
                
                data = self.keys.index(key)
                i = data // BOARD_COLS
                j = data % BOARD_COLS
                
                # Check if position is empty
                if self.state.data[i, j] != 0:
                    print("Position already occupied. Choose another.")
                    continue
                
                return i, j, self.symbol
            except (ValueError, IndexError):
                print("Invalid input. Please try again.")


def train(epochs: int, print_every_n: int = 500) -> Tuple[Player, Player]:
    """Train two AI players through self-play.
    
    Args:
        epochs: Number of games to play.
        print_every_n: Print statistics every N epochs.
        
    Returns:
        tuple: (player1, player2) trained players.
    """
    if epochs <= 0:
        raise ValueError("Epochs must be positive")
    
    player1 = Player(step_size=DEFAULT_STEP_SIZE, epsilon=TRAINING_EPSILON)
    player2 = Player(step_size=DEFAULT_STEP_SIZE, epsilon=TRAINING_EPSILON)
    judger = Judger(player1, player2)
    
    player1_wins = 0.0
    player2_wins = 0.0
    
    for i in range(1, epochs + 1):
        winner = judger.play(print_state=False)
        if winner == 1:
            player1_wins += 1
        elif winner == -1:
            player2_wins += 1
        
        if i % print_every_n == 0:
            logger.info(
                f"Epoch {i}, Player 1 winrate: {player1_wins / i:.2%}, "
                f"Player 2 winrate: {player2_wins / i:.2%}"
            )
        
        player1.backup()
        player2.backup()
        judger.reset()
    
    return player1, player2


def compete(turns: int = 1000) -> None:
    """Test trained players against each other.
    
    Args:
        turns: Number of games to play.
    """
    if turns <= 0:
        raise ValueError("Turns must be positive")
    
    player1 = Player(epsilon=EVAL_EPSILON)
    player2 = Player(epsilon=EVAL_EPSILON)
    judger = Judger(player1, player2)
    
    player1.load_policy()
    player2.load_policy()
    
    player1_wins = 0.0
    player2_wins = 0.0
    
    for _ in range(turns):
        winner = judger.play()
        if winner == 1:
            player1_wins += 1
        elif winner == -1:
            player2_wins += 1
        judger.reset()
    
    logger.info(
        f"{turns} games: Player 1 win rate {player1_wins / turns:.2%}, "
        f"Player 2 win rate {player2_wins / turns:.2%}, "
        f"Ties {(turns - player1_wins - player2_wins) / turns:.2%}"
    )


def play_interactive() -> None:
    """Play interactive game with human vs trained AI.
    
    Note: AI goes second and plays as player 2 (symbol -1).
    For a zero-sum game with optimal play, AI should guarantee at least a tie.
    """
    logger.info(
        "Starting interactive game. You are Player 1 (marked with *), "
        "AI is Player 2 (marked with x)."
    )
    
    while True:
        try:
            player1 = HumanPlayer()
            player2 = Player(epsilon=EVAL_EPSILON)
            judger = Judger(player1, player2)
            player2.load_policy()
            
            winner = judger.play(print_state=False)
            
            if winner == player2.symbol:
                logger.info("You lose!")
            elif winner == player1.symbol:
                logger.info("You win!")
            else:
                logger.info("It is a tie!")
            
            play_again = input("Play again? (y/n): ").lower()
            if play_again != 'y':
                break
        except KeyboardInterrupt:
            logger.info("\nGame interrupted.")
            break
        except Exception as e:
            logger.error(f"Error during game: {e}")
            break


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == 'train':
            epochs = int(sys.argv[2]) if len(sys.argv) > 2 else int(1e5)
            logger.info(f"Training for {epochs} epochs...")
            train(epochs)
            logger.info("Training complete. Playing test matches...")
            compete(int(1e3))
        elif command == 'play':
            play_interactive()
        else:
            print("Usage: python tic_tac_toe_improved.py [train|play] [epochs]")
        
    else:
        # Default: train and compete
        logger.info("Training for 100000 epochs...")
        train(int(1e5))
        logger.info("Competing for 1000 games...")
        compete(int(1e3))
        logger.info("Training complete!.")
        logger.info("Playingy")
        play_interactive()
