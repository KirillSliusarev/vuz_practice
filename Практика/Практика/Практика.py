import math
import random

import pygame
from pygame import mixer

# Intialize the pygame
pygame.init()
clock = pygame.time.Clock()
# create the screen
screen = pygame.display.set_mode((694, 768))

# Background
background = pygame.image.load('background.jpg')

# Caption and Icon
pygame.display.set_caption("Parasha")
icon = pygame.image.load('ufo.png')
pygame.display.set_icon(icon)
fps=60

# Player
playerImg = pygame.image.load('player.png')
playerX = 595
playerY = 685
playerX_change = 0
playerY_change = 0

def player(x, y):
    screen.blit(playerImg, (x, y))

# Game Loop
running = True
while running:
    clock.tick(fps)
    # RGB = Red, Green, Blue
    screen.fill((0, 0, 0))
    # Background Image
    screen.blit(background, (0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False



        # if keystroke is pressed check whether its right or left
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                playerX_change = -3
            if event.key == pygame.K_RIGHT:
                playerX_change = 3
            if event.key == pygame.K_UP:
                playerY_change = -3
            if event.key == pygame.K_DOWN:
                playerY_change = 3

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT or event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                playerX_change = 0
                playerY_change = 0

    # 5 = 5 + -0.1 -> 5 = 5 - 0.1
    # 5 = 5 + 0.1

    playerX += playerX_change
    if playerX <= 0:
        playerX = 0
    elif playerX >= 662:
        playerX = 662
    
    playerY += playerY_change
    if playerY <= 0:
        playerY = 0
    elif playerY >= 736:
        playerY = 736


    player(playerX, playerY)
    pygame.display.update()