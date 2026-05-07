import importlib.util
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from ament_index_python.packages import get_package_share_directory


def load_game_logic():
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / 'logic' / 'game_logic.py',
        Path.cwd() / 'src' / 'logic' / 'game_logic.py',
    ]
    try:
        share_dir = Path(get_package_share_directory('planning'))
        candidates.insert(0, share_dir / 'logic' / 'game_logic.py')
    except Exception:
        pass

    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location('game_logic', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    raise FileNotFoundError('Could not find project/src/logic/game_logic.py')


game_logic = load_game_logic()


class GameManager(Node):
    def __init__(self):
        super().__init__('game_manager')

        self.physical_board_rows = int(
            self.declare_parameter('physical_board_rows', 10).value
        )
        self.physical_board_cols = int(
            self.declare_parameter('physical_board_cols', 12).value
        )
        self.playable_row_offset = int(
            self.declare_parameter('playable_row_offset', 1).value
        )
        self.playable_col_offset = int(
            self.declare_parameter('playable_col_offset', 1).value
        )
        self.board_rows = int(
            self.declare_parameter(
                'board_rows',
                self.physical_board_rows - 2 * self.playable_row_offset
            ).value
        )
        self.board_cols = int(
            self.declare_parameter(
                'board_cols',
                self.physical_board_cols - 2 * self.playable_col_offset
            ).value
        )
        self.robot_player = int(
            self.declare_parameter('robot_player', 2).value
        )
        self.human_player = int(
            self.declare_parameter('human_player', 1).value
        )
        self.process_node = self.declare_parameter(
            'process_node',
            '/process_pointcloud'
        ).value
        self.start_service = self.declare_parameter(
            'start_service',
            '/start_robot_move'
        ).value
        self.use_click_input = bool(
            self.declare_parameter('use_click_input', True).value
        )

        self.engine = game_logic.BlokusEngine(
            rows=self.board_rows,
            cols=self.board_cols
        )
        self.robot_inventory = game_logic.get_mega_inventory()
        self.human_inventory = game_logic.get_mega_inventory()
        self.turn_count = {self.human_player: 0, self.robot_player: 0}

        self.param_client = self.create_client(
            SetParameters,
            f'{self.process_node}/set_parameters'
        )
        self.start_client = self.create_client(
            Trigger,
            self.start_service
        )

        self.get_logger().info(
            'Game manager ready. Enter human moves as: '
            'piece row col [rotation_index], or q to quit.'
        )
        if self.use_click_input:
            self.get_logger().info(
                'Click input is enabled. A board window will open for each '
                'human move if Tk is available.'
            )
        self.get_logger().info(
            f'Physical board is {self.physical_board_rows}x'
            f'{self.physical_board_cols}; playable game board is '
            f'{self.board_rows}x{self.board_cols}. The red 1x1 corner '
            'blocks are used only to contextualize the grid in space.'
        )

    def run_gamestate(self):
        while rclpy.ok():
            self.print_board()
            user_text = self.get_human_move_text()
            if user_text is None:
                break
            if user_text.lower() in ('q', 'quit', 'exit'):
                break
            if not self.apply_human_move(user_text):
                continue

            move = self.choose_robot_move()
            if move is None:
                self.get_logger().info('Robot has no legal move.')
                continue

            if self.send_robot_target(move):
                self.apply_robot_move(move)

    def get_human_move_text(self):
        if not self.use_click_input:
            return input('human move> ').strip()

        try:
            move_text = self.open_click_move_window()
        except Exception as exc:
            self.get_logger().warn(
                f'Click input unavailable ({exc}); falling back to terminal.'
            )
            return input('human move> ').strip()

        if move_text is None:
            return None
        return move_text

    def open_click_move_window(self):
        import tkinter as tk
        from tkinter import ttk

        result = {'move': None}
        selected = {'row': None, 'col': None}
        piece_names = list(self.human_inventory.keys())
        if not piece_names:
            self.get_logger().info('Human has no remaining pieces.')
            return 'q'

        root = tk.Tk()
        root.title('Human Move')
        root.resizable(False, False)

        header = ttk.Frame(root, padding=8)
        header.grid(row=0, column=0, sticky='ew')

        ttk.Label(header, text='Piece').grid(row=0, column=0, padx=4)
        piece_var = tk.StringVar(value=piece_names[0])
        piece_menu = ttk.OptionMenu(header, piece_var, piece_names[0], *piece_names)
        piece_menu.grid(row=0, column=1, padx=4)

        ttk.Label(header, text='Rotation').grid(row=0, column=2, padx=4)
        rotation_var = tk.IntVar(value=0)
        rotation_spin = ttk.Spinbox(
            header,
            from_=0,
            to=7,
            width=4,
            textvariable=rotation_var
        )
        rotation_spin.grid(row=0, column=3, padx=4)

        status_var = tk.StringVar(value='Click an origin cell.')
        ttk.Label(header, textvariable=status_var).grid(row=1, column=0, columnspan=4)

        grid_frame = ttk.Frame(root, padding=8)
        grid_frame.grid(row=1, column=0)

        buttons = {}

        def cell_color(row, col):
            value = self.engine.board[row, col]
            if value == self.human_player:
                return '#f4a6a6'
            if value == self.robot_player:
                return '#9ec5fe'
            return '#ffffff'

        def redraw():
            for (row, col), button in buttons.items():
                if row == selected['row'] and col == selected['col']:
                    button.configure(bg='#ffe082', text=f'{row},{col}')
                else:
                    button.configure(bg=cell_color(row, col), text='')

        def select_cell(row, col):
            selected['row'] = row
            selected['col'] = col
            status_var.set(f'Selected row={row}, col={col}')
            redraw()

        for row in range(self.board_rows):
            for col in range(self.board_cols):
                button = tk.Button(
                    grid_frame,
                    width=4,
                    height=2,
                    relief='solid',
                    borderwidth=1,
                    command=lambda r=row, c=col: select_cell(r, c)
                )
                button.grid(row=row, column=col)
                buttons[(row, col)] = button

        redraw()

        footer = ttk.Frame(root, padding=8)
        footer.grid(row=2, column=0, sticky='ew')

        def submit():
            row = selected['row']
            col = selected['col']
            if row is None or col is None:
                status_var.set('Pick a board cell first.')
                return

            piece = piece_var.get()
            rotation = rotation_var.get()
            result['move'] = f'{piece} {row} {col} {rotation}'
            root.destroy()

        def cancel():
            result['move'] = 'q'
            root.destroy()

        ttk.Button(footer, text='Use Move', command=submit).grid(row=0, column=0, padx=4)
        ttk.Button(footer, text='Quit', command=cancel).grid(row=0, column=1, padx=4)

        root.protocol('WM_DELETE_WINDOW', cancel)
        root.mainloop()
        return result['move']

    def apply_human_move(self, user_text):
        parts = user_text.split()
        if len(parts) not in (3, 4):
            self.get_logger().warn(
                'Use: piece row col [rotation_index], example: 1x2 0 0 0'
            )
            return False

        name = parts[0]
        if name not in self.human_inventory:
            self.get_logger().warn(f'Unknown human piece: {name}')
            return False

        try:
            row = int(parts[1])
            col = int(parts[2])
            rotation_index = int(parts[3]) if len(parts) == 4 else 0
        except ValueError:
            self.get_logger().warn('row, col, and rotation_index must be ints')
            return False

        variants = self.engine.get_transformations(self.human_inventory[name])
        coords = variants[rotation_index % len(variants)]
        first_move = self.turn_count[self.human_player] == 0

        if not self.engine.is_legal(
            self.human_player,
            coords,
            row,
            col,
            first_move=first_move
        ):
            self.get_logger().warn('Illegal human move')
            return False

        self.engine.place(self.human_player, coords, row, col)
        self.human_inventory.pop(name)
        self.turn_count[self.human_player] += 1
        return True

    def choose_robot_move(self):
        first_move = self.turn_count[self.robot_player] == 0
        moves = self.engine.get_legal_moves(
            self.robot_player,
            self.robot_inventory,
            first_move=first_move
        )
        row_aligned_moves = [
            move for move in moves
            if len({c for _, c in move['coords']}) == 1
        ]
        if not row_aligned_moves:
            return None

        def score_move(move):
            size_score = len(move['coords']) * 10
            row, col = move['origin']
            dist_score = 100 - (abs(row - 0) + abs(col - 0))
            return size_score + dist_score

        return max(row_aligned_moves, key=score_move)

    def apply_robot_move(self, move):
        row, col = move['origin']
        self.engine.place(self.robot_player, move['coords'], row, col)
        self.robot_inventory.pop(move['name'], None)
        self.turn_count[self.robot_player] += 1
        self.get_logger().info(
            f"Robot move: {move['name']} at row={row}, col={col}, "
            f"coords={move['coords']}"
        )

    def base_piece_name(self, name):
        return str(name).split('-', 1)[0]

    def send_robot_target(self, move):
        row, col = move['origin']
        base_name = self.base_piece_name(move['name'])
        target_cells = [
            (row + drow, col + dcol)
            for drow, dcol in move['coords']
        ]
        target_cell_text = ';'.join(
            f'{cell_row},{cell_col}'
            for cell_row, cell_col in target_cells
        )

        # Send the centroid of the piece cells so the robot places the block
        # at the middle of the span rather than at the origin corner.
        centroid_row = sum(r for r, _ in target_cells) / len(target_cells)
        centroid_col = sum(c for _, c in target_cells) / len(target_cells)

        # Determine whether the piece spans along the col axis (all same col,
        # different rows) or the row axis (all same row, different cols).
        piece_yaw_along_col = len({c for _, c in move['coords']}) == 1

        if not self.param_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error(
                f'Parameter service not available: {self.process_node}'
            )
            return False

        request = SetParameters.Request()
        request.parameters = [
            self.make_float_param('place_row', centroid_row),
            self.make_float_param('place_col', centroid_col),
            self.make_string_param('robot_target_cells', target_cell_text),
            self.make_string_param('expected_robot_shape', base_name),
            self.make_bool_param('piece_yaw_along_col', piece_yaw_along_col),
            self.make_bool_param('target_is_set', True),
        ]

        future = self.param_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        if future.result() is None:
            self.get_logger().error('Failed to set robot target parameters')
            return False

        ok = all(result.successful for result in future.result().results)
        if ok:
            self.get_logger().info(
                f'Sent robot target to perception: '
                f'centroid=({centroid_row:.2f},{centroid_col:.2f}), '
                f'yaw_along_col={piece_yaw_along_col}, '
                f'cells={target_cell_text}'
            )
            self.get_logger().info(
                f"Stage one blue {base_name} block in the pickup area "
                f"({move['name']}) "
                f"for target cells {target_cell_text}."
            )
            input('Press Enter to start the robot move...')
            return self.start_robot_move()

        self.get_logger().error('Perception rejected target parameters')
        return False

    def start_robot_move(self):
        if not self.start_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                f'Start service not available: {self.start_service}. '
                'Start the cube_grasp node with `ros2 run planning main`, '
                'or launch the stack that includes planning main, before '
                'pressing Enter in game_manager.'
            )
            return False

        future = self.start_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        result = future.result()
        if result is None:
            self.get_logger().error('Robot move start call timed out')
            return False

        if result.success:
            self.get_logger().info(result.message)
            return True

        self.get_logger().error(result.message)
        return False

    def make_float_param(self, name, value):
        param = Parameter()
        param.name = name
        param.value = ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE,
            double_value=value
        )
        return param

    def make_string_param(self, name, value):
        param = Parameter()
        param.name = name
        param.value = ParameterValue(
            type=ParameterType.PARAMETER_STRING,
            string_value=value
        )
        return param

    def make_bool_param(self, name, value):
        param = Parameter()
        param.name = name
        param.value = ParameterValue(
            type=ParameterType.PARAMETER_BOOL,
            bool_value=value
        )
        return param

    def print_board(self):
        rows = []
        for r in range(self.engine.rows):
            rows.append(' '.join(str(v) for v in self.engine.board[r]))
        self.get_logger().info('\n' + '\n'.join(rows))


def main(args=None):
    rclpy.init(args=args)
    node = GameManager()
    try:
        node.run_gamestate()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
