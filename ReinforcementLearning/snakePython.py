import random
import curses

def window_setup():
    # Set up the window
    window = curses.initscr()
    curses.curs_set(0)
    window_height, window_width = window.getmaxyx()
    window.keypad(1)
    window.timeout(150)  # Set the timeout for getch() to make the game run at a reasonable speed

    return window, window_height, window_width

def draw_snake(window, snake):
    for y, x in snake:
        window.addch(y, x, curses.ACS_BLOCK)

def draw_food(window, food):
    window.addch(food[0], food[1], curses.ACS_DIAMOND)

def move_snake(snake, direction):
    head = snake[0]
    new_head = [head[0] + direction[0], head[1] + direction[1]]
    snake.insert(0, new_head)
    return snake

def check_collision(snake, window_height, window_width):
    head = snake[0]
    if (
        head[0] in [0, window_height - 1]
        or head[1] in [0, window_width - 1]
        or head in snake[1:]
    ):
        return True
    return False

def generate_food(window_height, window_width):
    return [random.randint(1, window_height - 2), random.randint(1, window_width - 2)]

def main(stdscr):
    window, window_height, window_width = window_setup()

    snake = [
        [window_height // 2, window_width // 2],
        [window_height // 2, window_width // 2 - 1],
        [window_height // 2, window_width // 2 - 2],
    ]

    food = generate_food(window_height, window_width)

    direction = [0, 1]  # Initial direction: right

    while True:
        key = window.getch()

        if key == curses.KEY_UP:
            direction = [-1, 0]
        elif key == curses.KEY_DOWN:
            direction = [1, 0]
        elif key == curses.KEY_LEFT:
            direction = [0, -1]
        elif key == curses.KEY_RIGHT:
            direction = [0, 1]

        # Move the snake
        snake = move_snake(snake, direction)

        # Check for collisions
        if check_collision(snake, window_height, window_width):
            break

        # Check if snake eats food
        if snake[0] == food:
            food = generate_food(window_height, window_width)
        else:
            tail = snake.pop()

        # Draw the snake and food
        window.clear()
        draw_snake(window, snake)
        draw_food(window, food)

        window.refresh()

    curses.endwin()
    print("Game Over!")


if __name__ == "__main__":
    curses.wrapper(main)
