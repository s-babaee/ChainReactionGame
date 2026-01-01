# LTR
# chain_reaction_mcts_gui.py
import tkinter as tk
from tkinter import messagebox
from collections import deque
import copy
import math
import random
import time
import threading

# ----- Game parameters -----
ROWS = 5
COLS = 5
THRESHOLD = 4

# ----- MCTS parameters (defaults) -----
DEFAULT_MCTS_ITERATIONS = 100
UCT_C = 1.4
ROLLOUT_DEPTH = 10
DEFAULT_TIME_LIMIT = 5  # seconds; if set will override iterations

# ----- Game logic (same rules) -----

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

# def check_winner_simple(board):
#     """Check winner ignoring moves_made condition (used for simulations)."""
#     has_pos = any(cell > 0 for row in board for cell in row)
#     has_neg = any(cell < 0 for row in board for cell in row)
#     if has_pos and not has_neg:
#         return 1
#     if has_neg and not has_pos:
#         return 2
#     return None

def check_winner_ui(board, moves_made):
    if moves_made < 2:
        return None

    has_pos = any(cell > 0 for row in board for cell in row)
    has_neg = any(cell < 0 for row in board for cell in row)

    if has_pos and not has_neg:
        return 1
    if has_neg and not has_pos:
        return 2
    return None

# ----- Heuristic used in rollouts -----

def heuristic_score(board, player):
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

# ----- MCTS Node and functions -----

class MCTSNode:
    def __init__(self, board, first_move_flags, player, parent=None, move=None):
        self.board = board
        self.first_move_flags = first_move_flags
        self.player = player
        self.parent = parent
        self.move = move

        self.children = []
        self.untried_moves = get_all_valid_moves(self.board, self.player, self.first_move_flags)
        self.visits = 0
        self.wins = 0.0

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def expand(self):
        move = self.untried_moves.pop(random.randrange(len(self.untried_moves)))
        b_copy = copy_board(self.board)
        f_copy = dict(self.first_move_flags)
        apply_move(b_copy, self.player, move[0], move[1], f_copy)
        next_player = 2 if self.player == 1 else 1
        child = MCTSNode(b_copy, f_copy, next_player, parent=self, move=move)
        self.children.append(child)
        return child

    def best_child(self, c_param=UCT_C):
        best_score = -float("inf")
        best_child = None
        for child in self.children:
            if child.visits == 0:
                uct = float("inf")
            else:
                exploitation = child.wins / child.visits
                exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
                uct = exploitation + exploration
            if uct > best_score:
                best_score = uct
                best_child = child
        return best_child

    def most_visited_child(self):
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)

# ----- Rollout policy & simulation -----

def rollout_policy(board, player, first_move_flags):
    moves = get_all_valid_moves(board, player, first_move_flags)
    if not moves:
        return None

    scored = []
    for (r, c) in moves:
        b_copy = copy_board(board)
        f_copy = dict(first_move_flags)
        apply_move(b_copy, player, r, c, f_copy)
        score = heuristic_score(b_copy, player) + random.random() * 0.01
        scored.append(((r, c), score))
    scored.sort(key=lambda x: x[1], reverse=True)
    if random.random() < 0.08:
        return random.choice(moves)
    return scored[0][0]

def check_winner_simple(board):
    """
    Simple winner check used by MCTS/simulations.
    Returns 1 if only positive cells remain, 2 if only negative cells remain, else None.
    This does NOT use moves_made.
    """
    has_pos = any(cell > 0 for row in board for cell in row)
    has_neg = any(cell < 0 for row in board for cell in row)

    if has_pos and not has_neg:
        return 1
    if has_neg and not has_pos:
        return 2
    return None


def simulate_random_playout(board, first_move_flags, player, max_depth=ROLLOUT_DEPTH):
    b = copy_board(board)
    f = dict(first_move_flags)
    current = player
    for _ in range(max_depth):
        winner = check_winner_simple(b)
        if winner is not None:
            return winner
        move = rollout_policy(b, current, f)
        if move is None:
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
    s1 = heuristic_score(b, 1)
    s2 = heuristic_score(b, 2)
    if s1 > s2:
        return 1
    elif s2 > s1:
        return 2
    else:
        return random.choice([1, 2])

# ----- MCTS search -----

def mcts_search(root_board, root_first_move_flags, root_player, iterations=DEFAULT_MCTS_ITERATIONS, time_limit=None, ai_player=2):
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

        # Simulation
        winner = simulate_random_playout(node.board, node.first_move_flags, node.player)

        # Backpropagation
        reward = 1.0 if winner == ai_player else 0.0
        temp = node
        while temp is not None:
            temp.visits += 1
            temp.wins += reward
            temp = temp.parent

    best_child = root.most_visited_child()
    if best_child is None:
        return None
    return best_child.move

def ai_choose_move(board, first_move_flags, ai_player, iterations=DEFAULT_MCTS_ITERATIONS, time_limit=DEFAULT_TIME_LIMIT):
    move = mcts_search(board, first_move_flags, ai_player, iterations=iterations, time_limit=time_limit, ai_player=ai_player)
    if move is None:
        moves = get_all_valid_moves(board, ai_player, first_move_flags)
        return random.choice(moves) if moves else None
    return move

# ----- GUI Application -----

class MCTSGameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chain Reaction — MCTS AI")

        self.board = create_board()
        self.first_flags = {1: False, 2: False}
        self.moves_made = 0

        # default settings
        self.human_player = 1
        self.current = 1
        self.ai_player = 2
        self.ai_iterations = DEFAULT_MCTS_ITERATIONS
        self.ai_time_limit = DEFAULT_TIME_LIMIT
        self.ai_thinking = False

        # UI layout
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(padx=8, pady=6)

        # controls
        ctrl_frame = tk.Frame(self.top_frame)
        ctrl_frame.pack(side=tk.LEFT, padx=6)

        tk.Label(ctrl_frame, text="You are:").grid(row=0, column=0, sticky="w")
        self.player_var = tk.IntVar(value=1)
        tk.Radiobutton(ctrl_frame, text="Player 1 (+)", variable=self.player_var, value=1, command=self.on_player_choice).grid(row=1, column=0, sticky="w")
        tk.Radiobutton(ctrl_frame, text="Player 2 (-)", variable=self.player_var, value=2, command=self.on_player_choice).grid(row=2, column=0, sticky="w")

        tk.Label(ctrl_frame, text="Who starts:").grid(row=3, column=0, sticky="w", pady=(6,0))
        self.start_var = tk.StringVar(value="me")
        tk.Radiobutton(ctrl_frame, text="Me", variable=self.start_var, value="me").grid(row=4, column=0, sticky="w")
        tk.Radiobutton(ctrl_frame, text="Computer", variable=self.start_var, value="computer").grid(row=5, column=0, sticky="w")

        tk.Label(ctrl_frame, text="MCTS iterations:").grid(row=6, column=0, sticky="w", pady=(6,0))
        self.iter_entry = tk.Entry(ctrl_frame, width=8)
        self.iter_entry.insert(0, str(DEFAULT_MCTS_ITERATIONS))
        self.iter_entry.grid(row=7, column=0, sticky="w")

        tk.Label(ctrl_frame, text="Time limit (s, optional):").grid(row=8, column=0, sticky="w", pady=(6,0))
        self.time_entry = tk.Entry(ctrl_frame, width=8)
        self.time_entry.insert(0, "")
        self.time_entry.grid(row=9, column=0, sticky="w")

        self.start_btn = tk.Button(ctrl_frame, text="Start / Restart", command=self.start_game)
        self.start_btn.grid(row=10, column=0, pady=(8,0))

        # board frame
        self.board_frame = tk.Frame(self.top_frame)
        self.board_frame.pack(side=tk.LEFT, padx=12)

        self.cell_buttons = [[None for _ in range(COLS)] for _ in range(ROWS)]
        for r in range(ROWS):
            for c in range(COLS):
                b = tk.Button(self.board_frame, text="", width=6, height=3,
                              command=lambda rr=r, cc=c: self.on_cell_click(rr, cc),
                              font=("Consolas", 12, "bold"))
                b.grid(row=r, column=c, padx=4, pady=4)
                self.cell_buttons[r][c] = b

        # status
        self.status_label = tk.Label(root, text="Press Start to begin", font=("Arial", 12))
        self.status_label.pack(pady=(6,8))

        # initialize
        self.start_game()

    def on_player_choice(self):
        self.human_player = self.player_var.get()

    def read_settings(self):
        # read iterations and time limit
        try:
            n = int(self.iter_entry.get())
            if n <= 0:
                n = DEFAULT_MCTS_ITERATIONS
        except:
            n = DEFAULT_MCTS_ITERATIONS
        self.ai_iterations = n

        try:
            t = float(self.time_entry.get())
            if t <= 0:
                t = None
        except:
            t = None
        self.ai_time_limit = t

    def start_game(self):
        # reset state
        self.board = create_board()
        self.first_flags = {1: False, 2: False}
        self.moves_made = 0

        self.on_player_choice()  # update human_player
        starter = self.start_var.get()
        if starter == "me":
            self.current = self.human_player
        else:
            self.current = 2 if self.human_player == 1 else 1

        self.ai_player = 2 if self.human_player == 1 else 1
        self.read_settings()
        self.ai_thinking = False
        self.update_ui()
        # If computer starts, trigger its move
        if self.current == self.ai_player:
            self.root_after_ai_move()

    def on_cell_click(self, r, c):
        if self.ai_thinking:
            return
        if self.current != self.human_player:
            return
        if not valid_move(self.board, self.current, r, c, self.first_flags):
            messagebox.showinfo("Invalid move", "You can only click a cell that currently belongs to you (or a zero cell for your first move).")
            return
        apply_move(self.board, self.current, r, c, self.first_flags)
        self.moves_made += 1
        self.update_ui()

        winner = check_winner_ui(self.board, self.moves_made)
        if winner:
            self.end_game(winner)
            return

        self.current = 2 if self.current == 1 else 1
        # if next is AI, schedule AI move
        if self.current == self.ai_player:
            self.root_after_ai_move()

    def root_after_ai_move(self):
        # start AI in another thread to avoid freezing UI
        if self.ai_thinking:
            return
        self.read_settings()
        self.ai_thinking = True
        self.update_ui()
        t = threading.Thread(target=self._ai_worker, daemon=True)
        t.start()

    def _ai_worker(self):
        # copy state
        board_copy = copy_board(self.board)
        flags_copy = dict(self.first_flags)
        player_to_move = self.current
        # run MCTS (can be time-limited)
        move = ai_choose_move(board_copy, flags_copy, player_to_move, iterations=self.ai_iterations, time_limit=self.ai_time_limit)
        # schedule applying move on main thread
        if move is None:
            # fallback: no move (shouldn't happen)
            self.root.after(0, lambda: self._ai_done(None))
        else:
            self.root.after(0, lambda: self._ai_done(move))

    def _ai_done(self, move):
        self.ai_thinking = False
        if move is None:
            messagebox.showinfo("AI", "AI found no valid move.")
            return
        r, c = move
        if not valid_move(self.board, self.current, r, c, self.first_flags):
            # in rare cases the board changed; attempt to pick any valid move
            moves = get_all_valid_moves(self.board, self.current, self.first_flags)
            if not moves:
                messagebox.showinfo("AI", "AI found no valid move.")
                return
            r, c = random.choice(moves)
        apply_move(self.board, self.current, r, c, self.first_flags)
        self.moves_made += 1
        self.update_ui()
        winner = check_winner_ui(self.board, self.moves_made)
        if winner:
            self.end_game(winner)
            return
        self.current = 2 if self.current == 1 else 1
        self.update_ui()

    def end_game(self, winner):
        self.update_ui()
        if winner == self.human_player:
            messagebox.showinfo("Game Over", "You win! 🎉")
        else:
            messagebox.showinfo("Game Over", "Computer wins. 💻")
        # after game end, lock UI until restart
        self.ai_thinking = False

    def cell_colors(self, val):
        if val == 0:
            return ("white", "black")
        if val > 0:
            # positive -> green shades
            t = min(abs(val), 4)
            shades = {1: "#A9E5AE", 2: "#74D06B", 3: "#3FB02F", 4: "#2B8A22"}
            return (shades.get(t, "#2B8A22"), "#001100")
        else:
            t = min(abs(val), 4)
            shades = {1: "#F7B0B0", 2: "#F07A7A", 3: "#E54C4C", 4: "#C92A2A"}
            return (shades.get(t, "#C92A2A"), "#2C0000")

    def update_ui(self):
        # update button texts/colors
        for r in range(ROWS):
            for c in range(COLS):
                val = self.board[r][c]
                btn = self.cell_buttons[r][c]
                text = "" if val == 0 else str(abs(val))
                bg, fg = self.cell_colors(val)
                btn.config(text=text, bg=bg, fg=fg)

        if self.ai_thinking:
            status = "AI thinking..."
        else:
            winner = check_winner_ui(self.board, self.moves_made)
            if winner:
                status = "Game over"
            else:
                if self.current == self.human_player:
                    status = f"Your turn (Player {self.human_player})"
                else:
                    status = f"Computer's turn (Player {self.ai_player})"

        self.status_label.config(text=status)

# ----- run app -----

def main():
    root = tk.Tk()
    app = MCTSGameGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
