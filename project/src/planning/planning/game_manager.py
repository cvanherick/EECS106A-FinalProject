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

        self.board_rows = int(self.declare_parameter('board_rows', 12).value)
        self.board_cols = int(self.declare_parameter('board_cols', 10).value)
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

    def run_gamestate(self):
        while rclpy.ok():
            self.print_board()
            user_text = input('human move> ').strip()
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

    def send_robot_target(self, move):
        row, col = move['origin']

        if not self.param_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error(
                f'Parameter service not available: {self.process_node}'
            )
            return False

        request = SetParameters.Request()
        request.parameters = [
            self.make_float_param('place_row', float(row)),
            self.make_float_param('place_col', float(col)),
        ]

        future = self.param_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        if future.result() is None:
            self.get_logger().error('Failed to set robot target parameters')
            return False

        ok = all(result.successful for result in future.result().results)
        if ok:
            self.get_logger().info(
                f'Sent robot target to perception: row={row}, col={col}'
            )
            self.get_logger().info(
                f"Stage one red {move['name']} block in the pickup area."
            )
            input('Press Enter to start the robot move...')
            return self.start_robot_move()

        self.get_logger().error('Perception rejected target parameters')
        return False

    def start_robot_move(self):
        if not self.start_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error(
                f'Start service not available: {self.start_service}'
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
