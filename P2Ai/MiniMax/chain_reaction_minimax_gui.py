# LTR
# chain_reaction_minimax_gui.py
import tkinter as tk
from tkinter import messagebox
from collections import deque
import copy
import math
import random
import threading

# ---------------- Game parameters ----------------
ROWS = 5
COLS = 5
THRESHOLD = 4
DEFAULT_AI_DEPTH = 13

# GUI layout
CELL_SIZE = 80
PADDING = 6
CANVAS_WIDTH = COLS * CELL_SIZE
CANVAS_HEIGHT = ROWS * CELL_SIZE
FONT = ("Consolas", 20, "bold")

# ---------------- Game logic (unchanged minimax engine) ----------------

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
    q = deque()
    sign = 1 if player == 1 else -1
    val = board[r][c]

    # First move: place +3 or -3
    if not first_move_flags[player]:
        board[r][c] = 3 * sign
        first_move_flags[player] = True
        if abs(board[r][c]) >= THRESHOLD:
            q.append((r, c, sign))
        return

    # Normal move
    if sign == 1:
        if val < 3:
            board[r][c] += 1
            if board[r][c] >= THRESHOLD:
                q.append((r, c, sign))
        else:
            # Convert adjacent negative cells
            for nr, nc in neighbors(r, c):
                if board[nr][nc] < 0:
                    board[nr][nc] = abs(board[nr][nc])
            q.append((r, c, sign))
    else:
        if val > -3:
            board[r][c] -= 1
            if board[r][c] <= -THRESHOLD:
                q.append((r, c, sign))
        else:
            # Convert adjacent positive cells
            for nr, nc in neighbors(r, c):
                if board[nr][nc] > 0:
                    board[nr][nc] = -abs(board[nr][nc])
            q.append((r, c, sign))

    # Handle explosions
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

def check_winner_ui(board, moves_made):
    """UI winner check: requires at least 2 moves before declaring winner."""
    if moves_made < 2:
        return None
    has_pos = any(cell > 0 for row in board for cell in row)
    has_neg = any(cell < 0 for row in board for cell in row)
    if has_pos and not has_neg:
        return 1
    if has_neg and not has_pos:
        return 2
    return None

def check_winner_simple(board):
    """Used inside minimax (no moves_made constraint)."""
    has_pos = any(cell > 0 for row in board for cell in row)
    has_neg = any(cell < 0 for row in board for cell in row)
    if has_pos and not has_neg:
        return 1
    if has_neg and not has_pos:
        return 2
    return None

def evaluate(board, ai_player):
    ai_sign = 1 if ai_player == 1 else -1
    score = 0
    unstable_bonus = 6
    center_bonus = 0.5
    for r in range(ROWS):
        for c in range(COLS):
            v = board[r][c]
            score += ai_sign * v
            if abs(v) == 3:
                score += ai_sign * unstable_bonus
            center_dist = abs(r - (ROWS - 1) / 2) + abs(c - (COLS - 1) / 2)
            if v != 0:
                score += ai_sign * (center_bonus * (2.5 - center_dist))
    return score

def minimax(board, first_move_flags, player, depth, alpha, beta, maximizing, ai_player):
    winner = check_winner_simple(board)
    if winner is not None:
        return (10**6 if winner == ai_player else -10**6, None)
    if depth == 0:
        return (evaluate(board, ai_player), None)

    moves = get_all_valid_moves(board, player, first_move_flags)
    if not moves:
        return (evaluate(board, ai_player), None)

    random.shuffle(moves)
    best_move = None
    if maximizing:
        max_eval = -math.inf
        for r, c in moves:
            b_copy = copy_board(board)
            f_copy = dict(first_move_flags)
            apply_move(b_copy, player, r, c, f_copy)
            eval_child, _ = minimax(
                b_copy, f_copy, 2 if player == 1 else 1,
                depth - 1, alpha, beta, False, ai_player
            )
            if eval_child > max_eval:
                max_eval = eval_child
                best_move = (r, c)
            alpha = max(alpha, eval_child)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for r, c in moves:
            b_copy = copy_board(board)
            f_copy = dict(first_move_flags)
            apply_move(b_copy, player, r, c, f_copy)
            eval_child, _ = minimax(
                b_copy, f_copy, 2 if player == 1 else 1,
                depth - 1, alpha, beta, True, ai_player
            )
            if eval_child < min_eval:
                min_eval = eval_child
                best_move = (r, c)
            beta = min(beta, eval_child)
            if beta <= alpha:
                break
        return min_eval, best_move

def ai_choose_move(board, first_move_flags, ai_player, depth=DEFAULT_AI_DEPTH):
    _, move = minimax(
        board, first_move_flags, ai_player,
        depth, -math.inf, math.inf, True, ai_player
    )
    if move is None:
        moves = get_all_valid_moves(board, ai_player, first_move_flags)
        return random.choice(moves) if moves else None
    return move

# ---------------- GUI (Canvas based, stable layout) ----------------

class MinimaxCanvasGUI:
    def __init__(self, root):
        self.root = root
        root.title("Chain Reaction — Minimax GUI (stable map)")

        # game state
        self.board = create_board()
        self.first_flags = {1: False, 2: False}
        self.moves_made = 0

        # settings
        self.human_player = 1
        self.ai_player = 2
        self.current = 1
        self.ai_depth = DEFAULT_AI_DEPTH

        # frames
        control = tk.Frame(root)
        control.pack(side=tk.LEFT, padx=8, pady=8, fill=tk.Y)

        canvas_frame = tk.Frame(root)
        canvas_frame.pack(side=tk.LEFT, padx=8, pady=8)

        # canvas
        self.canvas = tk.Canvas(canvas_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white", highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # controls
        tk.Label(control, text="You are:").pack(anchor="w")
        self.player_var = tk.IntVar(value=1)
        tk.Radiobutton(control, text="Player 1 (+)", variable=self.player_var, value=1, command=self.on_player_change).pack(anchor="w")
        tk.Radiobutton(control, text="Player 2 (-)", variable=self.player_var, value=2, command=self.on_player_change).pack(anchor="w")

        tk.Label(control, text="Who starts:").pack(anchor="w", pady=(8,0))
        self.start_var = tk.StringVar(value="me")
        tk.Radiobutton(control, text="Me", variable=self.start_var, value="me").pack(anchor="w")
        tk.Radiobutton(control, text="Computer", variable=self.start_var, value="computer").pack(anchor="w")

        tk.Label(control, text="AI depth:").pack(anchor="w", pady=(8,0))
        self.depth_entry = tk.Entry(control, width=8)
        self.depth_entry.insert(0, str(DEFAULT_AI_DEPTH))
        self.depth_entry.pack(anchor="w")

        self.start_btn = tk.Button(control, text="Start / Restart", command=self.start_game)
        self.start_btn.pack(fill="x", pady=(12,6))

        self.status_label = tk.Label(control, text="Press Start", fg="blue")
        self.status_label.pack(pady=(6,0))

        # draw initial board
        self.draw_board()
        self.start_game()

    def on_player_change(self):
        self.human_player = self.player_var.get()

    def read_settings(self):
        try:
            d = int(self.depth_entry.get())
            if d < 1:
                d = DEFAULT_AI_DEPTH
        except:
            d = DEFAULT_AI_DEPTH
        self.ai_depth = d

    def start_game(self):
        self.board = create_board()
        self.first_flags = {1: False, 2: False}
        self.moves_made = 0
        self.on_player_change()
        self.read_settings()
        starter = self.start_var.get()
        self.current = self.human_player if starter == "me" else (2 if self.human_player == 1 else 1)
        self.ai_player = 2 if self.human_player == 1 else 1
        self.draw_board()
        self.update_status()
        if self.current == self.ai_player:
            self.root.after(200, self.start_ai_move)

    def canvas_to_cell(self, x, y):
        c = int(x // CELL_SIZE)
        r = int(y // CELL_SIZE)
        if 0 <= r < ROWS and 0 <= c < COLS:
            return r, c
        return None, None

    def on_canvas_click(self, event):
        if self.current != self.human_player:
            return
        r, c = self.canvas_to_cell(event.x, event.y)
        if r is None:
            return
        if not valid_move(self.board, self.current, r, c, self.first_flags):
            messagebox.showinfo("Invalid move", "You can only click a valid cell (your cell or zero on first move).")
            return
        apply_move(self.board, self.current, r, c, self.first_flags)
        self.moves_made += 1
        self.draw_board()
        winner = check_winner_ui(self.board, self.moves_made)
        if winner:
            self.end_game(winner)
            return
        self.current = 2 if self.current == 1 else 1
        if self.current == self.ai_player:
            self.root.after(80, self.start_ai_move)

    def _ai_worker(self, board_copy, flags_copy, player_to_move, depth, callback):
        move = ai_choose_move(board_copy, flags_copy, player_to_move, depth=depth)
        self.root.after(0, lambda: callback(move))

    def start_ai_move(self):
        if self.current != self.ai_player:
            return
        self.read_settings()
        self.status_label.config(text="AI thinking...", fg="orange")
        board_copy = copy_board(self.board)
        flags_copy = dict(self.first_flags)
        t = threading.Thread(target=self._ai_worker, args=(board_copy, flags_copy, self.current, self.ai_depth, self.finish_ai_move), daemon=True)
        t.start()

    def finish_ai_move(self, move):
        self.status_label.config(text="")
        if move is None:
            moves = get_all_valid_moves(self.board, self.current, self.first_flags)
            if not moves:
                messagebox.showinfo("AI", "AI found no valid move.")
                return
            move = random.choice(moves)
        r, c = move
        if not valid_move(self.board, self.current, r, c, self.first_flags):
            moves = get_all_valid_moves(self.board, self.current, self.first_flags)
            if not moves:
                messagebox.showinfo("AI", "AI found no valid move.")
                return
            r, c = random.choice(moves)
        apply_move(self.board, self.current, r, c, self.first_flags)
        self.moves_made += 1
        self.draw_board()
        winner = check_winner_ui(self.board, self.moves_made)
        if winner:
            self.end_game(winner)
            return
        self.current = 2 if self.current == 1 else 1
        self.update_status()

    def end_game(self, winner):
        if winner == self.human_player:
            messagebox.showinfo("Game Over", "You win! 🎉")
        else:
            messagebox.showinfo("Game Over", "Computer wins. 💻")
        self.status_label.config(text="Game over")

    def cell_colors(self, val):
        if val == 0:
            return ("#ffffff", "#000000")
        t = min(abs(val), 4)
        if val > 0:
            shades = {1: "#A9E5AE", 2: "#74D06B", 3: "#3FB02F", 4: "#2B8A22"}
            return (shades[t], "#001100")
        else:
            shades = {1: "#F7B0B0", 2: "#F07A7A", 3: "#E54C4C", 4: "#C92A2A"}
            return (shades[t], "#2C0000")

    def draw_board(self):
        self.canvas.delete("all")
        # grid
        for r in range(ROWS):
            for c in range(COLS):
                x1 = c * CELL_SIZE
                y1 = r * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                val = self.board[r][c]
                bg, fg = self.cell_colors(val)
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=bg, outline="#666666")
                if val != 0:
                    self.canvas.create_text(x1 + CELL_SIZE/2, y1 + CELL_SIZE/2, text=str(abs(val)), font=FONT, fill=fg)
        # column letters
        for c in range(COLS):
            self.canvas.create_text(c*CELL_SIZE + CELL_SIZE/2, CANVAS_HEIGHT - 10, text=chr(ord('A')+c), font=("Arial",10), fill="#333333")
        # row numbers (optional draw left side)
        # we keep canvas compact; row numbers can be shown in status if needed
        self.update_status()

    def update_status(self):
        winner = check_winner_ui(self.board, self.moves_made)
        if winner:
            self.status_label.config(text=f"Player {winner} wins!", fg="blue")
        else:
            if self.current == self.human_player:
                self.status_label.config(text=f"Your turn (Player {self.human_player})", fg="green")
            else:
                self.status_label.config(text=f"Computer thinking (Player {self.ai_player})", fg="orange")

# ---------------- Run ----------------

def main():
    root = tk.Tk()
    app = MinimaxCanvasGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
