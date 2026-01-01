# LTR
# chain_reaction_mcts.py
from collections import deque
import copy
import math
import random
import time

# ----- Game parameters -----
ROWS = 5
COLS = 5
THRESHOLD = 4

# ----- MCTS parameters (tune these) -----
MCTS_ITERATIONS = 3000   # more -> stronger but slower
UCT_C = 1.4              # exploration constant
ROLLOUT_DEPTH = 60       # max moves in a rollout to avoid infinite games
TIME_LIMIT = None        # seconds; if set, overrides MCTS_ITERATIONS

# ----- Game logic (same as your rules) -----

def create_board(rows=ROWS, cols=COLS):
    return [[0 for _ in range(cols)] for _ in range(rows)]

def neighbors(r, c, rows=ROWS, cols=COLS):
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc

def copy_board(board):
    return [row.copy() for row in board]

def valid_move(board, player, r, c, first_move_flags):
    if not (0 <= r < ROWS and 0 <= c < COLS):
        return False
    val = board[r][c]
    if not first_move_flags[player]:
        return val == 0
    return val > 0 if player == 1 else val < 0

def apply_move(board, player, r, c, first_move_flags):
    """
    Apply a move in-place. Maintains first_move_flags and performs chain reactions.
    """
    q = deque()
    sign = 1 if player == 1 else -1
    val = board[r][c]

    # First move (place +3 or -3)
    if not first_move_flags[player]:
        board[r][c] = 3 * sign
        first_move_flags[player] = True
        if abs(board[r][c]) >= THRESHOLD:
            q.append((r, c, sign))
        return

    # Normal move
    if sign == 1:
        if val < 3:
            board[r][c] = val + 1
            if board[r][c] >= THRESHOLD:
                q.append((r, c, sign))
        else:
            # convert adjacent negatives
            for nr, nc in neighbors(r, c):
                if board[nr][nc] < 0:
                    board[nr][nc] = abs(board[nr][nc])
            q.append((r, c, sign))
    else:
        if val > -3:
            board[r][c] = val - 1
            if board[r][c] <= -THRESHOLD:
                q.append((r, c, sign))
        else:
            # convert adjacent positives
            for nr, nc in neighbors(r, c):
                if board[nr][nc] > 0:
                    board[nr][nc] = -abs(board[nr][nc])
            q.append((r, c, sign))

    # handle chain reactions
    while q:
        cr, cc, s = q.popleft()
        board[cr][cc] = 0
        for nr, nc in neighbors(cr, cc):
            new_abs = abs(board[nr][nc]) + 1
            board[nr][nc] = s * new_abs
            if abs(board[nr][nc]) >= THRESHOLD:
                q.append((nr, nc, s))

def get_all_valid_moves(board, player, first_move_flags):
    moves = []
    for r in range(ROWS):
        for c in range(COLS):
            if valid_move(board, player, r, c, first_move_flags):
                moves.append((r, c))
    return moves

def check_winner_simple(board):
    """Check winner ignoring moves_made condition (used for simulations)."""
    has_pos = any(cell > 0 for row in board for cell in row)
    has_neg = any(cell < 0 for row in board for cell in row)
    if has_pos and not has_neg:
        return 1
    if has_neg and not has_pos:
        return 2
    return None

# ----- Utility: a simple evaluation heuristic used in rollouts -----

def heuristic_score(board, player):
    """
    Simple heuristic to guide rollouts:
    - Sum of values (signed) in favor of 'player'
    - Bonus for unstable cells (abs == 3)
    - Preference for center control
    """
    sign = 1 if player == 1 else -1
    score = 0.0
    for r in range(ROWS):
        for c in range(COLS):
            v = board[r][c]
            score += sign * v
            if abs(v) == 3:
                score += sign * 6.0
            center_dist = abs(r - (ROWS - 1) / 2) + abs(c - (COLS - 1) / 2)
            if v != 0:
                score += sign * (0.5 * (2.5 - center_dist))
    return score

# ----- MCTS structures and functions -----

class MCTSNode:
    def __init__(self, board, first_move_flags, player, parent=None, move=None):
        self.board = board                # board state (list of lists)
        self.first_move_flags = first_move_flags  # dict {1:bool,2:bool}
        self.player = player              # player to move at this node
        self.parent = parent
        self.move = move                  # move that led to this node (r,c) or None for root

        self.children = []
        self.untried_moves = get_all_valid_moves(self.board, self.player, self.first_move_flags)
        self.visits = 0
        self.wins = 0.0   # count of simulations where AI (caller) won — stored from AI perspective in backprop

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def expand(self):
        """Expand by taking one untried move and returning the new child node."""
        move = self.untried_moves.pop(random.randrange(len(self.untried_moves)))
        # copy state
        b_copy = copy_board(self.board)
        f_copy = dict(self.first_move_flags)
        apply_move(b_copy, self.player, move[0], move[1], f_copy)
        next_player = 2 if self.player == 1 else 1
        child = MCTSNode(b_copy, f_copy, next_player, parent=self, move=move)
        self.children.append(child)
        return child

    def best_child(self, c_param=UCT_C):
        """Return child with highest UCT score (wins/visits + exploration)."""
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                uct = float("inf")
            else:
                exploitation = child.wins / child.visits
                exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
                uct = exploitation + exploration
            choices_weights.append((uct, child))
        return max(choices_weights, key=lambda x: x[0])[1]

    def most_visited_child(self):
        """Used to select final move: child with highest visits."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)

# ----- Rollout policy (heuristic-guided playout) -----

def rollout_policy(board, player, first_move_flags):
    """
    Choose a move during rollout:
    - Prefer moves that create/trigger unstable cells or convert opponent cells
    - Fall back to random choice if no clear preference
    """
    moves = get_all_valid_moves(board, player, first_move_flags)
    if not moves:
        return None

    scored = []
    for (r, c) in moves:
        # quick simulate the move on a shallow copy to estimate impact
        b_copy = copy_board(board)
        f_copy = dict(first_move_flags)
        apply_move(b_copy, player, r, c, f_copy)
        # heuristic: prefer higher heuristic_score for the acting player
        score = heuristic_score(b_copy, player)
        # also give small random tie-breaker
        score += random.random() * 0.01
        scored.append(((r, c), score))
    # pick best-scored move with probability; occasionally explore
    scored.sort(key=lambda x: x[1], reverse=True)
    # small epsilon to explore
    if random.random() < 0.08:
        return random.choice(moves)
    return scored[0][0]

def simulate_random_playout(board, first_move_flags, player, max_depth=ROLLOUT_DEPTH):
    """
    Simulate a game until terminal or depth limit using rollout_policy.
    Returns winner (1 or 2) or None.
    """
    b = copy_board(board)
    f = dict(first_move_flags)
    current = player
    for _ in range(max_depth):
        winner = check_winner_simple(b)
        if winner is not None:
            return winner
        move = rollout_policy(b, current, f)
        if move is None:
            # no valid moves: treat as draw-like; evaluate by heuristic
            # choose winner by heuristic score
            s1 = heuristic_score(b, 1)
            s2 = heuristic_score(b, 2)
            if s1 > s2:
                return 1
            elif s2 > s1:
                return 2
            else:
                return random.choice([1, 2])
        apply_move(b, current, move[0], move[1], f)
        current = 2 if current == 1 else 1
    # depth reached: evaluate by heuristic
    s1 = heuristic_score(b, 1)
    s2 = heuristic_score(b, 2)
    if s1 > s2:
        return 1
    elif s2 > s1:
        return 2
    else:
        return random.choice([1, 2])

# ----- MCTS main routine -----

def mcts_search(root_board, root_first_move_flags, root_player, iterations=MCTS_ITERATIONS, time_limit=None, ai_player=2):
    """
    Run MCTS from given state and return the best move for the root_player.
    ai_player is the agent we measure wins for (usually the computer's player id).
    """
    root = MCTSNode(copy_board(root_board), dict(root_first_move_flags), root_player, parent=None, move=None)
    start_time = time.time()

    it = 0
    while True:
        if time_limit is not None and (time.time() - start_time) >= time_limit:
            break
        if time_limit is None and iterations is not None and it >= iterations:
            break
        it += 1

        # Selection
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        # Expansion
        if not node.is_fully_expanded():
            node = node.expand()

        # Simulation / Rollout
        # simulate from node's state; current player is node.player
        winner = simulate_random_playout(node.board, node.first_move_flags, node.player)

        # Backpropagation: update visits and wins (wins counted for ai_player)
        reward = 1.0 if winner == ai_player else 0.0
        temp = node
        while temp is not None:
            temp.visits += 1
            temp.wins += reward
            temp = temp.parent

    # choose best child by most visits
    best_child = root.most_visited_child()
    if best_child is None:
        return None
    return best_child.move

# ----- Integration: ai_choose_move using MCTS -----

def ai_choose_move(board, first_move_flags, ai_player, iterations=MCTS_ITERATIONS, time_limit=TIME_LIMIT):
    """
    Wrapper to call MCTS and return (r,c) for ai_player.
    If time_limit is provided (seconds) it will run until time elapses.
    """
    move = mcts_search(board, first_move_flags, ai_player, iterations=iterations, time_limit=time_limit, ai_player=ai_player)
    if move is None:
        # fallback: pick random valid move
        moves = get_all_valid_moves(board, ai_player, first_move_flags)
        return random.choice(moves) if moves else None
    return move

# ----- Console UI (single-player) -----

def print_board(board):
    print("\n     " + "   ".join(chr(ord('A') + i) for i in range(len(board[0]))))
    for r, row in enumerate(board):
        line = []
        for val in row:
            line.append(f"{val:3d}" if val != 0 else "  .")
        print(f"{r+1:>2} " + " ".join(line))
    print()

def parse_move(move_str):
    move = move_str.strip().upper().replace(",", " ").split()
    if not move:
        return None
    if len(move) == 1 and len(move[0]) >= 2 and move[0][0].isalpha():
        col = ord(move[0][0]) - ord('A')
        try:
            row = int(move[0][1:]) - 1
        except:
            return None
        return row, col
    if len(move) >= 2:
        a, b = move[0], move[1]
        if a.isalpha():
            col = ord(a[0]) - ord('A')
            try:
                row = int(b) - 1
            except:
                return None
            return row, col
        else:
            try:
                row = int(a) - 1
                col = int(b) - 1
            except:
                return None
            return row, col
    return None

def main():
    board = create_board()
    moves_made = 0
    first_move_flags = {1: False, 2: False}

    print("Chain Reaction (5x5) — Single Player vs MCTS AI")
    print("Rules: first move place ±3 on a zero cell; P1 plays positive cells; P2 plays negative cells.")
    print("Explosion occurs when absolute value reaches 4.\n")

    while True:
        choice = input("Choose your player (1 = positive, 2 = negative) [default 1]: ").strip()
        if choice == "":
            human = 1
            break
        if choice in ("1", "2"):
            human = int(choice)
            break
        print("Enter 1 or 2.")

    starter = input("Who starts? (me/computer) [default me]: ").strip().lower() or "me"
    current = human if starter == "me" else (2 if human == 1 else 1)
    print(f"\nYou are Player {human}. Game starts. (MCTS iterations = {MCTS_ITERATIONS})\n")

    while True:
        print_board(board)
        winner = check_winner_simple(board)
        if winner:
            print("You win!" if winner == human else "Computer wins!")
            break

        if current == human:
            move_input = input(f"Player {current} (you) move > ")
            parsed = parse_move(move_input)
            if not parsed:
                print("Invalid input. Example: A1 or 2 3")
                continue
            r, c = parsed
            if not valid_move(board, current, r, c, first_move_flags):
                print("Invalid move.")
                continue
            apply_move(board, current, r, c, first_move_flags)
            moves_made += 1
        else:
            print("Computer is thinking (MCTS)...")
            # you can pass time_limit or iterations override here if desired
            move = ai_choose_move(board, first_move_flags, current, iterations=MCTS_ITERATIONS, time_limit=TIME_LIMIT)
            if move is None:
                print("Computer found no valid move.")
                break
            r, c = move
            print(f"Computer plays {chr(ord('A') + c)}{r + 1}")
            apply_move(board, current, r, c, first_move_flags)
            moves_made += 1

        current = 2 if current == 1 else 1

if __name__ == "__main__":
    main()
