import math
import random
import time

import cv2
import mediapipe as mp
import numpy as np
import pygame

import assets
import configs
from objects.background import Background
from objects.bird import Bird
from objects.column import Column
from objects.floor import Floor
from objects.gameover_message import GameOverMessage
from objects.gamestart_message import GameStartMessage
from objects.retry import RetryButton
from objects.score import Score

MIN_DISTANCE = 0.07
MAX_DISTANCE = 0.5
MAX_JUMP_INTERVAL = 0.01
MIN_JUMP_INTERVAL = 2


pygame.init()


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

screen = pygame.display.set_mode(
    (configs.SCREEN_WIDTH, configs.SCREEN_HEIGHT))

pygame.display.set_caption("Hand-Flappy Bird")

img = pygame.image.load('assets/icons/red_bird.png')
pygame.display.set_icon(img)

clock = pygame.time.Clock()
column_create_event = pygame.USEREVENT
running = True
gameover = False
gamestarted = False

assets.load_sprites()
assets.load_audios()

sprites = pygame.sprite.LayeredUpdates()

retry_button_image = assets.get_sprite("retry")
retry_button = RetryButton(sprites)
retry_button.kill()


def reset_game():
    global gameover, gamestarted, sprites, bird, game_start_message, score, last_jump_time
    gameover = False
    gamestarted = False
    sprites.empty()
    bird, game_start_message, score = create_sprites()
    last_jump_time = time.time()
    retry_button.kill()


def create_sprites():
    Background(0, sprites)
    Background(1, sprites)
    Floor(0, sprites)
    Floor(1, sprites)
    bird = Bird(sprites)
    bird.reset_position()
    return bird, GameStartMessage(sprites), Score(sprites)


bird, game_start_message, score = create_sprites()


jump_interval = MAX_JUMP_INTERVAL
last_jump_time = time.time()


cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == column_create_event:
                Column(sprites)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE and gameover and not gamestarted:
                    gamestarted = True
                    game_start_message.kill()
                    pygame.time.set_timer(column_create_event, 5000)
                if event.key == pygame.K_SPACE and not gamestarted and not gameover:
                    gamestarted = True
                    game_start_message.kill()
                    pygame.time.set_timer(column_create_event, 5000)
                if event.key == pygame.K_ESCAPE and gameover:
                    gameover = False
                    gamestarted = False
                    sprites.empty()
                    bird, game_start_message, score = create_sprites()

        success, image = cap.read()
        if success:
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            distance = 0
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    distance = math.hypot(
                        index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)

                    image_height, image_width, _ = image.shape
                    thumb_px = (int(thumb_tip.x * image_width),
                                int(thumb_tip.y * image_height))
                    index_px = (int(index_tip.x * image_width),
                                int(index_tip.y * image_height))

                    cv2.line(image, thumb_px, index_px, (255, 0, 0), 2)

                    midpoint_x = int((thumb_px[0] + index_px[0]) / 2)
                    midpoint_y = int((thumb_px[1] + index_px[1]) / 2)

                    cv2.putText(image, f"Distance: {distance:.2f}", (midpoint_x, midpoint_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(
                            color=(255, 255, 255), thickness=10, circle_radius=2),
                        mp_drawing_styles.get_default_hand_connections_style())

            if distance > MIN_DISTANCE:
                jump_interval = np.interp(distance, [MIN_DISTANCE, MAX_DISTANCE], [
                                          MIN_JUMP_INTERVAL, MAX_JUMP_INTERVAL])
            else:
                jump_interval = 10

            cv2.imshow('MediaPipe Hands', image)

        current_time = time.time()
        if gamestarted and not gameover and (current_time - last_jump_time >= jump_interval):
            bird.jump()
            last_jump_time = current_time

        screen.fill(0)
        sprites.draw(screen)

        if gamestarted and not gameover:
            sprites.update()

        if bird.check_collision(sprites) and not gameover:
            gameover = True
            gamestarted = False
            RetryButton(sprites)
            GameOverMessage(sprites)
            pygame.time.set_timer(column_create_event, 0)
            assets.play_audio("hit")

        if gameover:
            retry_button.update(reset_game)

        for sprite in sprites:
            if type(sprite) is Column and sprite.is_passed():
                score.value += 1
                assets.play_audio("point")

        pygame.display.flip()
        clock.tick(configs.FPS)

        if cv2.waitKey(5) & 0xFF == 27:
            break


cap.release()
cv2.destroyAllWindows()
pygame.quit()
