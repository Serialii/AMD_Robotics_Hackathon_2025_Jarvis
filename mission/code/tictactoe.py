import subprocess
import sys
import cv2
import numpy as np
from typing import Optional, Tuple, List

class TicTacToe:
    def __init__(self, replay_base_path: str = ".", robot_type: str = "so101_follower", 
                 robot_port: str = "/dev/ttyACM1", robot_id: str = "my_awesome_follower_arm",
                 dataset_repo: str = "Abubakar17/hi", camera_id: int = 0):
        """
        Initialize the Tic-Tac-Toe game controller.
        
        Args:
            replay_base_path: Path where replay episodes are stored
            robot_type: Type of robot
            robot_port: Serial port for robot connection
            robot_id: Robot identifier
            dataset_repo: Dataset repository ID
            camera_id: Camera device ID (default 0 for primary camera)
        """
        self.replay_base = replay_base_path
        self.robot_type = robot_type
        self.robot_port = robot_port
        self.robot_id = robot_id
        self.dataset_repo = dataset_repo
        
        # Board state: 0 = empty, 1 = human, -1 = robot
        self.board = [0] * 9
        self.human_player = 1
        self.robot_player = -1
        
        # Camera setup
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        # Board detection parameters
        self.board_detected = False
        self.board_corners = None
        self.last_detected_board = [0] * 9
    
    def close_camera(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def preprocess_frame(self, frame):
        """Preprocess frame for board detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
        return thresh
    
    def detect_board(self, frame):
        """
        Detect tic-tac-toe board in the frame.
        Returns corners of the board if detected.
        """
        thresh = self.preprocess_frame(frame)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour (likely the board)
        largest = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        
        # Check if we have 4 corners (quadrilateral)
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)

            return corners
        
        return None
    
    def warp_board(self, frame, corners):
        """Warp board to a top-down perspective."""
        h, w = frame.shape[:2]
        
        # Sort corners: top-left, top-right, bottom-right, bottom-left
        center = corners.mean(axis=0)
        sorted_corners = sorted(corners, key=lambda p: (np.arctan2(p[1] - center[1], p[0] - center[0]), 
                                                        np.linalg.norm(p - center)))
        sorted_corners = np.array([sorted_corners[i] for i in [1, 0, 3, 2]], dtype=np.float32)  # Make sure it's float32
        
        # Destination points (300x300 board)
        board_size = 300
        dst_points = np.array([
            [0, 0],
            [board_size, 0],
            [board_size, board_size],
            [0, board_size]
        ], dtype=np.float32)
        
        # Get perspective transform
        M = cv2.getPerspectiveTransform(sorted_corners, dst_points)
        warped = cv2.warpPerspective(frame, M, (board_size, board_size))
        
        return warped, sorted_corners

    
    def detect_marks(self, warped_board):
        """
        Detect X and O marks in each cell of the warped board.
        Returns board state: 0 = empty, 1 = human (X), -1 = robot (O)
        """
        board_state = [0] * 9
        cell_size = warped_board.shape[0] // 3
        
        gray = cv2.cvtColor(warped_board, cv2.COLOR_BGR2GRAY)
        
        for row in range(3):
            for col in range(3):
                # Extract cell
                x1, x2 = row * cell_size, (row + 1) * cell_size
                y1, y2 = col * cell_size, (col + 1) * cell_size
                cell = gray[y1:y2, x1:x2]
                
                # Invert colors to make marks darker than background
                cell_inv = cv2.bitwise_not(cell)
                
                # Apply threshold to get binary image of marks
                _, cell_thresh = cv2.threshold(cell_inv, 100, 255, cv2.THRESH_BINARY)
                
                # Count dark pixels (marks) vs light pixels (empty)
                dark_pixels = np.sum(cell_thresh > 0)
                total_pixels = cell_size * cell_size
                mark_ratio = dark_pixels / total_pixels
                
                # If sufficient pixels are marked, it's not empty
                if mark_ratio > 0.08:  # Threshold for detecting a mark
                    # Find contours to distinguish X from O
                    contours, _ = cv2.findContours(cell_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if len(contours) > 0:
                        # X typically has 2 separate strokes, O has 1 continuous shape
                        # Count the number of significant contours
                        significant_contours = [c for c in contours if cv2.contourArea(c) > 20]
                        
                        if len(significant_contours) >= 2:
                            board_state[row * 3 + col] = self.human_player  # X
                        else:
                            board_state[row * 3 + col] = self.robot_player  # O
        
        return board_state
    
    def capture_and_detect_board(self):
        """
        Capture frame from camera and detect current board state.
        Returns the detected board state, or None if board not detected.
        """
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame")
            return None
        
        # Display the frame for debugging
        display_frame = frame.copy()
        
        # Detect board corners
        corners = self.detect_board(frame)
        
        if corners is None:
            cv2.imshow("Board Detection", display_frame)
            cv2.waitKey(1)
            return None
        
        # Warp board to top-down view
        warped, _ = self.warp_board(frame, corners)
        
        # Detect marks in each cell
        detected_board = self.detect_marks(warped)
        
        # Draw grid on warped board for visualization
        display_warped = warped.copy()
        cell_size = warped.shape[0] // 3
        for i in range(1, 3):
            cv2.line(display_warped, (i * cell_size, 0), (i * cell_size, warped.shape[0]), (0, 255, 0), 2)
            cv2.line(display_warped, (0, i * cell_size), (warped.shape[1], i * cell_size), (0, 255, 0), 2)
        
        # Add detected marks to display
        symbols = {0: " ", 1: "X", -1: "O"}
        for i in range(9):
            row, col = i // 3, i % 3
            y = row * cell_size + cell_size // 2
            x = col * cell_size + cell_size // 2
            symbol = symbols[detected_board[i]]
            if symbol != " ":
                cv2.putText(display_warped, symbol, (y - 10, x + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Warped Board Detection", display_warped)
        cv2.imshow("Board Detection", display_frame)
        cv2.waitKey(1)
        
        return detected_board
    
    def wait_for_human_move(self, timeout_ms: int = 60000):
        """
        Wait for human to make a move by capturing board changes.
        
        Args:
            timeout_ms: Timeout in milliseconds
            
        Returns:
            Position (0-8) of the new move, or -1 if timeout
        """
        print("Waiting for human move...")
        import time
        start_time = time.time()
        timeout_s = timeout_ms / 1000.0
        
        prev_board = self.board.copy()
        
        while time.time() - start_time < timeout_s:
            detected = self.capture_and_detect_board()
            
            if detected is None:
                continue
            
            # Check if board changed from previous state
            differences = []
            for i in range(9):
                if prev_board[i] != detected[i] and prev_board[i] == 0:
                    differences.append(i)
            
            # If exactly one new mark appeared, it's the human's move
            if len(differences) == 1:
                move_pos = differences[0]
                if detected[move_pos] == self.human_player:
                    print(f"Detected human move at position {move_pos}")
                    self.board = detected
                    return move_pos
            
            # Small delay to avoid excessive checking
            import time as time_module
            time_module.sleep(0.1)
        
        print("Timeout waiting for human move")
        return -1
    
    def print_board(self):
        """Print the current board state."""
        symbols = {0: " ", 1: "X", -1: "O"}
        print("\n")
        for i in range(3):
            row = [symbols[self.board[i*3 + j]] for j in range(3)]
            print(f" {row[0]} | {row[1]} | {row[2]} ")
            if i < 2:
                print("-----------")
        print("\n")
    
    def get_empty_cells(self) -> List[int]:
        """Return list of empty cell indices."""
        return [i for i in range(9) if self.board[i] == 0]
    
    def check_winner(self, board: List[int]) -> int:
        """
        Check if there's a winner.
        Returns: 1 if human wins, -1 if robot wins, 0 if no winner.
        """
        winning_combos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        
        for combo in winning_combos:
            if board[combo[0]] == board[combo[1]] == board[combo[2]] != 0:
                return board[combo[0]]
        
        return 0
    
    def is_game_over(self, board: List[int]) -> bool:
        """Check if game is over."""
        return self.check_winner(board) != 0 or len([i for i in board if i == 0]) == 0
    
    def minimax(self, board: List[int], depth: int, is_maximizing: bool) -> int:
        """Minimax algorithm."""
        winner = self.check_winner(board)
        
        if winner == self.robot_player:
            return 10 - depth
        elif winner == self.human_player:
            return depth - 10
        elif len([i for i in board if i == 0]) == 0:
            return 0
        
        if is_maximizing:
            max_score = float('-inf')
            for i in range(9):
                if board[i] == 0:
                    board[i] = self.robot_player
                    score = self.minimax(board, depth + 1, False)
                    board[i] = 0
                    max_score = max(score, max_score)
            return max_score
        else:
            min_score = float('inf')
            for i in range(9):
                if board[i] == 0:
                    board[i] = self.human_player
                    score = self.minimax(board, depth + 1, True)
                    board[i] = 0
                    min_score = min(score, min_score)
            return min_score
    
    def get_best_move(self) -> int:
        """Get best move using minimax."""
        best_score = float('-inf')
        best_move = None
        
        for i in range(9):
            if self.board[i] == 0:
                self.board[i] = self.robot_player
                score = self.minimax(self.board, 0, False)
                self.board[i] = 0
                
                if score > best_score:
                    best_score = score
                    best_move = i
        
        return best_move
    
    def play_move(self, position: int) -> bool:
        """Execute a robot move."""
        if position < 0 or position > 8 or self.board[position] != 0:
            print(f"Invalid move: position {position}")
            return False
        
        self.board[position] = self.robot_player
        
        row = position // 3
        col = position % 3
        episode_name = f"{row}{col}"
        
        cmd = [
            "lerobot-replay",
            f"--robot.type={self.robot_type}",
            f"--robot.port={self.robot_port}",
            f"--robot.id={self.robot_id}",
            f"--dataset.repo_id={self.dataset_repo}/{episode_name}",
            f"--dataset.episode=0"
        ]
        
        print(f"Playing move at position {position} ({row}, {col})")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
            print(f"Move executed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error executing replay: {e}")
            return False
        except subprocess.TimeoutExpired:
            print(f"Replay command timed out")
            return False
    
    def play_game(self):
        """Main game loop with computer vision."""
        print("Tic-Tac-Toe with LeRobot - CV Edition")
        print("Positions: 0 1 2 / 3 4 5 / 6 7 8")
        print("\nPositioning camera to view the board...")
        
        # Calibrate by detecting initial board
        print("Detecting board... (press any key once board is visible)")
        initial_board = None
        while initial_board is None:
            initial_board = self.capture_and_detect_board()
        
        print("Board detected!")
        
        human_first = input("Should human play first? (y/n): ").lower() == 'y'
        self.board = [0] * 9
        
        while not self.is_game_over(self.board):
            self.print_board()
            
            if human_first:
                # Wait for human move
                move_pos = self.wait_for_human_move()
                if move_pos == -1:
                    print("Timeout waiting for move")
                    break
                
                if self.is_game_over(self.board):
                    break
                
                # Robot move
                best_move = self.get_best_move()
                print(f"Robot move: {best_move}")
                if not self.play_move(best_move):
                    print("Failed to execute robot move!")
                    break
            else:
                # Robot move first
                best_move = self.get_best_move()
                print(f"Robot move: {best_move}")
                if not self.play_move(best_move):
                    print("Failed to execute robot move!")
                    break
                
                if self.is_game_over(self.board):
                    break
                
                # Wait for human move
                move_pos = self.wait_for_human_move()
                if move_pos == -1:
                    print("Timeout waiting for move")
                    break
        
        self.print_board()
        winner = self.check_winner(self.board)
        if winner == self.robot_player:
            print("Robot wins! 🤖")
        elif winner == self.human_player:
            print("Human wins! 🎉")
        else:
            print("It's a draw!")
        
        self.close_camera()


if __name__ == "__main__":
    replay_path = "replays"
    
    try:
        game = TicTacToe(
            replay_base_path=replay_path,
            robot_type="so101_follower",
            robot_port="/dev/ttyACM1",
            robot_id="my_awesome_follower_arm",
            dataset_repo="Abubakar17",
            camera_id=2
        )
        
        game.play_game()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            game.close_camera()
        except:
            pass