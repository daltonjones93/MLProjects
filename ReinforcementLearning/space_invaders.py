import pygame
import random

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 400
PLAYER_SIZE = 50
ENEMY_SIZE = 30
BULLET_SIZE = 5
PLAYER_SPEED = 5
ENEMY_SPEED = 2
BULLET_SPEED = 5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Create the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Space Invaders")

# Player
player = pygame.Rect(WIDTH // 2 - PLAYER_SIZE // 2, HEIGHT - 2 * PLAYER_SIZE, PLAYER_SIZE, PLAYER_SIZE)
player_color = WHITE

# Enemies
enemies = []
for _ in range(5):
    enemy = pygame.Rect(random.randint(0, WIDTH - ENEMY_SIZE), random.randint(50, 150), ENEMY_SIZE, ENEMY_SIZE)
    enemies.append(enemy)

# Bullets
bullets = []

# Clock to control the frame rate
clock = pygame.time.Clock()

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and player.left > 0:
        player.x -= PLAYER_SPEED
    if keys[pygame.K_RIGHT] and player.right < WIDTH:
        player.x += PLAYER_SPEED

    # Shoot bullets
    if keys[pygame.K_SPACE]:
        bullet = pygame.Rect(player.centerx - BULLET_SIZE // 2, player.top, BULLET_SIZE, BULLET_SIZE)
        bullets.append(bullet)

    # Move bullets
    bullets = [bullet for bullet in bullets if bullet.y > 0]
    for bullet in bullets:
        bullet.y -= BULLET_SPEED

    # Move enemies
    for enemy in enemies:
        enemy.y += ENEMY_SPEED
        if enemy.y > HEIGHT:
            enemy.y = 0
            enemy.x = random.randint(0, WIDTH - ENEMY_SIZE)

    # Check for collisions
    for bullet in bullets:
        for enemy in enemies:
            if bullet.colliderect(enemy):
                bullets.remove(bullet)
                enemies.remove(enemy)
                enemy.x = random.randint(0, WIDTH - ENEMY_SIZE)
                enemy.y = 0

    if any(player.colliderect(enemy) for enemy in enemies):
        running = False

    # Draw everything
    screen.fill(BLACK)
    pygame.draw.rect(screen, player_color, player)
    for enemy in enemies:
        pygame.draw.rect(screen, RED, enemy)
    for bullet in bullets:
        pygame.draw.rect(screen, WHITE, bullet)

    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()
