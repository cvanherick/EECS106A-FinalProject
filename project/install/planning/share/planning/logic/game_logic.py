import numpy as np
import random

class BlokusEngine:
    def __init__(self, rows=10, cols=12):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)
        self.dsu = {1: {}, 2: {}}

    def get_transformations(self, piece_coords):
        if not piece_coords: return []
        coords = np.array(piece_coords)
        coords -= coords.min(axis=0)
        grid = np.zeros((coords[:,0].max()+1, coords[:,1].max()+1), dtype=int)
        for r, c in coords: grid[r, c] = 1
        
        variants = set()
        curr = grid
        for _ in range(2):
            for _ in range(4):
                curr = np.rot90(curr)
                new_c = tuple(sorted([(r, c) for r in range(curr.shape[0]) for c in range(curr.shape[1]) if curr[r, c]]))
                variants.add(new_c)
            curr = np.fliplr(curr)
        return [list(v) for v in variants]

    def is_legal(self, player_id, coords, origin_r, origin_c, first_move=False):
        has_corner = False
        actual_coords = []
        for dr, dc in coords:
            r, c = origin_r + dr, origin_c + dc
            if not (0 <= r < self.rows and 0 <= c < self.cols) or self.board[r, c] != 0:
                return False
            for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.board[nr, nc] == player_id:
                    return False
            for nr, nc in [(r-1,c-1),(r-1,c+1),(r+1,c-1),(r+1,c+1)]:
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.board[nr, nc] == player_id:
                    has_corner = True
            actual_coords.append((r, c))
        if first_move:
            return any((r==0 and c==0) or (r==self.rows-1 and c==self.cols-1) for r, c in actual_coords)
        return has_corner

    def place(self, player_id, coords, origin_r, origin_c):
        for dr, dc in coords:
            r, c = origin_r + dr, origin_c + dc
            self.board[r, c] = player_id
            self._update_dsu(player_id, r, c)

    def _update_dsu(self, p_id, r, c):
        self.dsu[p_id][(r, c)] = (r, c)
        for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if (nr, nc) in self.dsu[p_id]:
                self._union(p_id, (r, c), (nr, nc))

    def _find(self, p_id, i):
        if self.dsu[p_id][i] == i: return i
        self.dsu[p_id][i] = self._find(p_id, self.dsu[p_id][i])
        return self.dsu[p_id][i]

    def _union(self, p_id, i, j):
        root_i, root_j = self._find(p_id, i), self._find(p_id, j)
        if root_i != root_j: self.dsu[p_id][root_i] = root_j

    def get_legal_moves(self, player_id, inventory, first_move=False):
        moves = []
        for name, shape in inventory.items():
            for trans in self.get_transformations(shape):
                for r in range(self.rows):
                    for c in range(self.cols):
                        if self.is_legal(player_id, trans, r, c, first_move):
                            moves.append({'name': name, 'coords': trans, 'origin': (r, c)})
        return moves

    def get_greedy_move(self, player_id, inventory, first_move=False):
        moves = self.get_legal_moves(player_id, inventory, first_move)
        if not moves: return None
        
        def score_move(move):
            # Priority 1: Piece size (len of coords)
            # Priority 2: Distance from their start (attacking the center)
            size_score = len(move['coords']) * 10
            r_orig, c_orig = move['origin']
            # If player 2 (robot), try to get to (0,0). If player 1, get to (rows, cols)
            dist_target = (0,0) if player_id == 2 else (self.rows-1, self.cols-1)
            dist_score = 100 - (abs(r_orig - dist_target[0]) + abs(c_orig - dist_target[1]))
            return size_score + dist_score

        return max(moves, key=score_move)

def get_mega_inventory():
    return {
        '1x2-a': [(0, 0), (0, 1)],
        '1x2-b': [(0, 0), (0, 1)],
        '1x3-a': [(0, 0), (0, 1), (0, 2)],
        '1x3-b': [(0, 0), (0, 1), (0, 2)],
        '1x4-a': [(0, 0), (0, 1), (0, 2), (0, 3)],
        '1x4-b': [(0, 0), (0, 1), (0, 2), (0, 3)],
    }
