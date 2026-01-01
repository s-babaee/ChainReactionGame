from collections import deque

ROWS = 5
COLS = 5
THRESHOLD = 4

def create_board(rows=ROWS, cols=COLS):
    return [[0 for _ in range(cols)] for _ in range(rows)]

def neighbors(r, c, rows=ROWS, cols=COLS):
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
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
    val = board[r][c]
    if not first_move_flags[player]:
        return val == 0
    if player == 1:
        return val > 0
    else:
        return val < 0

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

def main():
    board = create_board()
    current = 1
    moves_made = 0
    first_move_flags = {1: False, 2: False}

    print("Chain Reaction (5x5) — Final Version")
    print("First move: place 3 or -3 on a zero cell")
    print("After that: Player 1 plays on positive cells, Player 2 on negative cells")
    print("An explosion happens when the absolute value reaches 4\n")

    while True:
        print_board(board)
        winner = check_winner(board, moves_made)
        if winner:
            print(f"\n🎉 Player {winner} wins! 🎉\n")
            break

        move_input = input(f"Player {current}'s move > ").strip()
        if move_input.lower() in ("q", "quit", "exit"):
            print("Game aborted.")
            break

        parsed = parse_move(move_input)
        if not parsed:
            print("Invalid move format. Example: A1 or 2 3")
            continue
        r, c = parsed

        if not valid_move(board, current, r, c, first_move_flags):
            if not first_move_flags[current]:
                print("Invalid move: the first move must be on a zero cell")
            else:
                print("Invalid move: you can only click a cell that already belongs to you")
            continue

        apply_move(board, current, r, c, first_move_flags)
        moves_made += 1

        winner = check_winner(board, moves_made)
        if winner:
            print_board(board)
            print(f"\n🎉 Player {winner} wins! 🎉\n")
            break

        current = 2 if current == 1 else 1

if __name__ == "__main__":
    main()
