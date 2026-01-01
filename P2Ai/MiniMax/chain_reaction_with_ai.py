from collections import deque
import copy
import math
import random

ROWS = 5
COLS = 5
THRESHOLD = 4
AI_DEPTH = 13  # MiniMax search depth (higher = stronger but slower)

def create_board(rows=ROWS, cols=COLS):
    return [[0 for _ in range(cols)] for _ in range(rows)]

def neighbors(r, c, rows=ROWS, cols=COLS):
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc

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

def check_winner_simple(board):
    has_pos = any(cell > 0 for row in board for cell in row)
    has_neg = any(cell < 0 for row in board for cell in row)

    if has_pos and not has_neg:
        return 1
    if has_neg and not has_pos:
        return 2
    return None

def copy_board(board):
    return [row.copy() for row in board]

def get_all_valid_moves(board, player, first_move_flags):
    moves = []
    for r in range(ROWS):
        for c in range(COLS):
            if valid_move(board, player, r, c, first_move_flags):
                moves.append((r, c))
    return moves

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

def ai_choose_move(board, first_move_flags, ai_player, depth=AI_DEPTH):
    _, move = minimax(
        board, first_move_flags, ai_player,
        depth, -math.inf, math.inf, True, ai_player
    )
    if move is None:
        moves = get_all_valid_moves(board, ai_player, first_move_flags)
        return random.choice(moves) if moves else None
    return move

def main():
    board = create_board()
    moves_made = 0
    first_move_flags = {1: False, 2: False}

    # print("Chain Reaction (5x5) — Single Player vs AI")
    # print("Rules:")
    # print("- First move: place +3 or -3 on a zero cell")
    # print("- Player 1 plays on positive cells")
    # print("- Player 2 plays on negative cells")
    # print("- Explosion occurs when absolute value reaches 4\n")

    while True:
        choice = input("Choose your player (1 = positive, 2 = negative) [default 1]: ").strip()
        if choice == "":
            human = 1
            break
        if choice in ("1", "2"):
            human = int(choice)
            break

    starter = input("Who starts? (me/computer) [default me]: ").strip().lower() or "me"
    current = human if starter == "me" else (2 if human == 1 else 1)

    print(f"\nYou are Player {human}. Game starts.\n")

    while True:
        print_board(board)
        winner = check_winner(board, moves_made)
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
            print("Computer is thinking...")
            r, c = ai_choose_move(board, first_move_flags, current)
            print(f"Computer plays {chr(ord('A') + c)}{r + 1}")
            apply_move(board, current, r, c, first_move_flags)
            moves_made += 1

        current = 2 if current == 1 else 1

if __name__ == "__main__":
    main()
