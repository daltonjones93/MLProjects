#include <iostream>
#include <ncurses.h>
#include <cstdlib>
#include <ctime>
#include <vector>

using namespace std;

// Function to generate random numbers in a given range
int random(int min, int max) {
    return rand() % (max - min + 1) + min;
}

// Function to check if the snake has collided with the walls or itself
bool isGameOver(int headX, int headY, const vector<int>& tailX, const vector<int>& tailY, int width, int height) {
    if (headX < 0 || headX >= width || headY < 0 || headY >= height) {
        return true;  // Collided with walls
    }

    for (size_t i = 0; i < tailX.size(); ++i) {
        if (headX == tailX[i] && headY == tailY[i]) {
            return true;  // Collided with itself
        }
    }

    return false;
}

int main() {
    // Set up the game board
    const int width = 20;
    const int height = 10;
    const char snakeChar = 'O';
    const char fruitChar = '*';

    // Initial position of the snake
    int headX = width / 2;
    int headY = height / 2;

    // Initial direction of the snake
    int directionX = 1;
    int directionY = 0;

    // Initial position of the fruit
    int fruitX = random(0, width - 1);
    int fruitY = random(0, height - 1);

    // Score and length of the snake
    int score = 0;
    vector<int> tailX;
    vector<int> tailY;

    // Initialize the ncurses library
    initscr();
    keypad(stdscr, TRUE);
    nodelay(stdscr, TRUE);
    noecho();
    curs_set(0);

    // Game loop
    while (true) {
        // Display the game board
        clear();

        for (int i = 0; i < width + 2; ++i) {
            printw("#");  // Top border
        }
        printw("\n");

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                if (j == 0) {
                    printw("#");  // Left border
                }

                if (i == headY && j == headX) {
                    printw("%c", snakeChar);  // Snake head
                } else if (i == fruitY && j == fruitX) {
                    printw("%c", fruitChar);  // Fruit
                } else {
                    bool isTailSegment = false;
                    for (size_t k = 0; k < tailX.size(); ++k) {
                        if (tailX[k] == j && tailY[k] == i) {
                            printw("%c", snakeChar);  // Snake tail segment
                            isTailSegment = true;
                        }
                    }

                    if (!isTailSegment) {
                        printw(" ");  // Empty space
                    }
                }

                if (j == width - 1) {
                    printw("#");  // Right border
                }
            }
            printw("\n");
        }

        for (int i = 0; i < width + 2; ++i) {
            printw("#");  // Bottom border
        }
        printw("\n");

        // Check for collisions and update the game state
        if (isGameOver(headX, headY, tailX, tailY, width, height)) {
            printw("Game Over! Your score: %d\n", score);
            break;
        }

        // Check if the snake has eaten the fruit
        if (headX == fruitX && headY == fruitY) {
            score += 10;

            // Generate a new fruit position
            fruitX = random(0, width - 1);
            fruitY = random(0, height - 1);

            // Increase the length of the snake
            tailX.push_back(0);
            tailY.push_back(0);
        }

        // Update the position of the snake's tail
        for (size_t i = tailX.size() - 1; i > 0; --i) {
            tailX[i] = tailX[i - 1];
            tailY[i] = tailY[i - 1];
        }

        // Update the position of the snake's head
        tailX[0] = headX;
        tailY[0] = headY;

        // Move the snake's head in the current direction
        headX += directionX;
        headY += directionY;

        // Get user input for changing the direction
        int key = getch();
        switch (key) {
            case KEY_UP:
                directionY = -1;
                directionX = 0;
                break;
            case KEY_DOWN:
                directionY = 1;
                directionX = 0;
                break;
            case KEY_LEFT:
                directionY = 0;
                directionX = -1;
                break;
            case KEY_RIGHT:
                directionY = 0;
                directionX = 1;
                break;
            case 'x':
                printw("Game Over! Your score: %d\n", score);
                endwin();
                return 0;
        }

        // Control the speed of the game
        // You may need to adjust this based on your system
        // and the speed at which the console updates
        napms(100);
        refresh();
    }

    // End the ncurses session
    endwin();

    return 0;
}
