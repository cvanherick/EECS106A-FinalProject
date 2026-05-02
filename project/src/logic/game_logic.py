import numpy as np

class BlokusLogic:
    def __init__(self, rows=10, cols=12):
        """
        Initialize the game board.
        State Representation:
        0 = Empty, 1 = Player 1 (Human), 2 = Player 2 (Robot)
        """
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        
    def get_piece_transformations(self, piece):
        """
        Takes a piece (list of (r, c) coordinates) and returns all 
        unique rotations (0, 90, 180, 270) and flips (horizontal/vertical).
        """
        # TODO: Implement matrix rotation and flipping logic here.
        # You can use numpy's rot90 and fliplr/flipud if you represent 
        # pieces as small 2D arrays instead of coordinate lists.
        pass

    def is_valid_move(self, player_id, piece_coords, origin_r, origin_c, is_first_turn=False):
        """
        The Umpire: Checks if a proposed move is entirely legal.
        piece_coords: List of relative (r, c) tuples, e.g., [(0,0), (0,1), (1,0)]
        origin_r, origin_c: Where on the board the piece's (0,0) coordinate is placed
        """
        corner_touch = False

        for r_offset, c_offset in piece_coords:
            r = origin_r + r_offset
            c = origin_c + c_offset

            # 1. Bounds Check
            if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
                return False 

            # 2. Collision Check (Square must be empty)
            if self.board[r, c] != 0:
                return False

            # 3. Edge Check (Negative Constraint)
            # Cannot share a flat edge with your own piece
            orthogonals = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            for adj_r, adj_c in orthogonals:
                if 0 <= adj_r < self.rows and 0 <= adj_c < self.cols:
                    if self.board[adj_r, adj_c] == player_id:
                        return False # Illegal edge connection

            # 4. Corner Check (Positive Constraint)
            # Must touch at least one of your own corners
            diagonals = [(r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)]
            for diag_r, diag_c in diagonals:
                if 0 <= diag_r < self.rows and 0 <= diag_c < self.cols:
                    if self.board[diag_r, diag_c] == player_id:
                        corner_touch = True

        # First turn exception: You don't have to touch a corner of your own piece,
        # but you usually have to touch a designated board corner.
        if is_first_turn:
            # TODO: Add logic to ensure the piece covers a specific board corner 
            # e.g., (0,0) or (self.rows-1, self.cols-1)
            return True

        return corner_touch

    def place_piece(self, player_id, piece_coords, origin_r, origin_c):
        """
        Applies the piece to the board. 
        Note: Always call is_valid_move() before calling this!
        """
        for r_offset, c_offset in piece_coords:
            r = origin_r + r_offset
            c = origin_c + c_offset
            self.board[r, c] = player_id

    def get_all_legal_moves(self, player_id, player_inventory, is_first_turn=False):
        """
        The Engine for the AI: Scans the board to find every single legal 
        move available for a given player based on their remaining pieces.
        """
        legal_moves = []
        
        # This is a brute-force approach. For optimization later, you only 
        # need to scan squares that are diagonal to the player's existing pieces.
        for r in range(self.rows):
            for c in range(self.cols):
                for piece in player_inventory:
                    transformations = self.get_piece_transformations(piece)
                    for trans in transformations:
                        if self.is_valid_move(player_id, trans, r, c, is_first_turn):
                            legal_moves.append({
                                'piece': trans,
                                'origin': (r, c)
                            })
                            
        return legal_moves

    def print_board(self):
        """Helper to visualize the board in the terminal."""
        print(self.board)