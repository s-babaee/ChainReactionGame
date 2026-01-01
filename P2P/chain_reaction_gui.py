import tkinter as tk
from tkinter import messagebox
from collections import deque

ROWS = 5
COLS = 5
THRESHOLD = 4

# --- Game logic functions (adapted from your script) ---

def create_board(rows=ROWS, cols=COLS):
    return [[0 for _ in range(cols)] for _ in range(rows)]

def neighbors(r, c, rows=ROWS, cols=COLS):
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc

def valid_move(board, player, r, c, first_move_flags):
    if not (0 <= r < ROWS and 0 <= c < COLS):
        return False
    val = board[r][c]
    if not first_move_flags[player]:
        return val == 0
    return val > 0 if player == 1 else val < 0

def apply_move(board, player, r, c, first_move_flags):
    rows, cols = len(board), len(board[0])
    q = deque()
    sign = 1 if player == 1 else -1
    val = board[r][c]

    if not first_move_flags[player]:
        board[r][c] = 3 * sign
        first_move_flags[player] = True
        if abs(board[r][c]) >= THRESHOLD:
            q.append((r, c, sign))
        return

    if sign == 1:
        if val < 3:
            board[r][c] = val + 1
            if board[r][c] >= THRESHOLD:
                q.append((r, c, sign))
        else:
            for nr, nc in neighbors(r, c, rows, cols):
                if board[nr][nc] < 0:
                    board[nr][nc] = abs(board[nr][nc])
            q.append((r, c, sign))
    else:
        if val > -3:
            board[r][c] = val - 1
            if board[r][c] <= -THRESHOLD:
                q.append((r, c, sign))
        else:
            for nr, nc in neighbors(r, c, rows, cols):
                if board[nr][nc] > 0:
                    board[nr][nc] = -abs(board[nr][nc])
            q.append((r, c, sign))

    while q:
        cr, cc, s = q.popleft()
        board[cr][cc] = 0
        for nr, nc in neighbors(cr, cc, rows, cols):
            new_abs = abs(board[nr][nc]) + 1
            board[nr][nc] = s * new_abs
            if abs(board[nr][nc]) >= THRESHOLD:
                q.append((nr, nc, s))

def check_winner(board, moves_made):
    if moves_made < 2:
        return None
    has_pos = any(cell > 0 for row in board for cell in row)
    has_neg = any(cell < 0 for row in board for cell in row)
    if has_pos and not has_neg:
        return 1
    if has_neg and not has_pos:
        return 2
    return None

# --- GUI Application ---

class ChainReactionGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chain Reaction (5x5)")
        self.resizable(False, False)

        # Game state
        self.board = create_board()
        self.first_move_flags = {1: False, 2: False}
        self.current = 1
        self.moves_made = 0

        # UI components
        self.cell_buttons = [[None for _ in range(COLS)] for _ in range(ROWS)]
        self.info_label = tk.Label(self, text="", font=("Helvetica", 12))
        self.info_label.grid(row=0, column=0, columnspan=COLS, pady=(6,0))

        board_frame = tk.Frame(self)
        board_frame.grid(row=1, column=0, columnspan=COLS, padx=8, pady=8)

        btn_width = 6
        btn_height = 3
        for r in range(ROWS):
            for c in range(COLS):
                b = tk.Button(board_frame, text=".", width=btn_width, height=btn_height,
                              command=lambda rr=r, cc=c: self.on_cell_click(rr, cc),
                              font=("Consolas", 14, "bold"))
                b.grid(row=r, column=c, padx=4, pady=4)
                self.cell_buttons[r][c] = b

        ctrl_frame = tk.Frame(self)
        ctrl_frame.grid(row=2, column=0, columnspan=COLS, pady=(0,8))

        self.restart_btn = tk.Button(ctrl_frame, text="Restart", command=self.restart)
        self.restart_btn.pack(side=tk.LEFT, padx=6)

        self.print_console_var = tk.IntVar(value=0)
        self.print_console_cb = tk.Checkbutton(ctrl_frame, text="Print to console", variable=self.print_console_var)
        self.print_console_cb.pack(side=tk.LEFT, padx=6)

        self.update_ui()

    def cell_color(self, value):
        if value == 0:
            return "#DDDDDD", "#000000"  # light gray background
        if value > 0:
            # green shades by magnitude
            t = min(abs(value), 4)
            # stronger -> darker green
            shades = {1: "#A9E5AE", 2: "#74D06B", 3: "#3FB02F", 4: "#2B8A22"}
            return shades.get(t, "#2B8A22"), "#001100"
        else:
            t = min(abs(value), 4)
            shades = {1: "#F7B0B0", 2: "#F07A7A", 3: "#E54C4C", 4: "#C92A2A"}
            return shades.get(t, "#C92A2A"), "#2C0000"

    def update_ui(self):
        for r in range(ROWS):
            for c in range(COLS):
                val = self.board[r][c]
                bg, fg = self.cell_color(val)
                btn = self.cell_buttons[r][c]
                text = "." if val == 0 else str(val)
                btn.config(text=text, bg=bg, fg=fg)
        self.info_label.config(text=f"Current: Player {self.current}    (Player1 = +, Player2 = -)")
        if self.print_console_var.get():
            self.print_board_console()

    def print_board_console(self):
        print("\n     " + "   ".join(chr(ord('A') + i) for i in range(COLS)))
        for r, row in enumerate(self.board):
            line = []
            for val in row:
                line.append(f"{val:3d}" if val != 0 else "  .")
            print(f"{r+1:>2} " + " ".join(line))
        print()

    def on_cell_click(self, r, c):
        if not valid_move(self.board, self.current, r, c, self.first_move_flags):
            messagebox.showinfo("Invalid move", "You can only click a cell that currently belongs to you "
                                                "or (for the first move) a zero cell.")
            return

        apply_move(self.board, self.current, r, c, self.first_move_flags)
        self.moves_made += 1
        self.update_ui()

        winner = check_winner(self.board, self.moves_made)
        if winner:
            self.update_ui()
            messagebox.showinfo("Game Over", f"Player {winner} wins!")
            return

        # switch player
        self.current = 2 if self.current == 1 else 1
        self.update_ui()

    def restart(self):
        if messagebox.askyesno("Restart", "Restart the game?"):
            self.board = create_board()
            self.first_move_flags = {1: False, 2: False}
            self.current = 1
            self.moves_made = 0
            self.update_ui()

def main():
    app = ChainReactionGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
