import pygame
import sys
import math
import random

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Player Ship class
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("player.png").convert()
        self.image.set_colorkey(WHITE)
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH // 2, HEIGHT // 2)
        self.angle = 0

    def rotate(self, angle_change):
        self.angle += angle_change
        self.image = pygame.transform.rotate(self.image, angle_change)
        self.rect = self.image.get_rect(center=self.rect.center)

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.rotate(5)
        if keys[pygame.K_RIGHT]:
            self.rotate(-5)
        if keys[pygame.K_UP]:
            self.move()

    def move(self):
        angle_rad = math.radians(self.angle)
        self.rect.x += 5 * math.cos(angle_rad)
        self.rect.y -= 5 * math.sin(angle_rad)

# Asteroid class
class Asteroid(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("asteroid.png").convert()
        self.image.set_colorkey(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(WIDTH - self.rect.width)
        self.rect.y = random.randrange(HEIGHT - self.rect.height)
        self.angle = random.randrange(360)

    def update(self):
        self.rotate(2)

    def rotate(self, angle_change):
        self.angle += angle_change
        self.image = pygame.transform.rotate(self.image, angle_change)
        self.rect = self.image.get_rect(center=self.rect.center)

# Initialize game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Asteroids Game")

# Load images
player_img = pygame.image.load("player.png").convert()
asteroid_img = pygame.image.load("asteroid.png").convert()

# Create sprites
all_sprites = pygame.sprite.Group()
asteroids = pygame.sprite.Group()

player = Player()
all_sprites.add(player)

for _ in range(5):
    asteroid = Asteroid()
    all_sprites.add(asteroid)
    asteroids.add(asteroid)

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update
    all_sprites.update()

    # Check for collisions
    hits = pygame.sprite.spritecollide(player, asteroids, False)
    if hits:
        print("Game Over!")
        running = False

    # Draw
    screen.fill(BLACK)
    all_sprites.draw(screen)

    # Flip the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(FPS)

# Quit Pygame
pygame.quit()
sys.exit()
