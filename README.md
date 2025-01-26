# Flappy Bird with Hand Gesture Control

## Overview

This project is a unique take on the classic Flappy Bird game, implemented in Pygame and enhanced with innovative hand gesture control using MediaPipe. The game challenges players to navigate a bird through a series of pipes, with the twist that the bird's jump frequency is controlled by the player's hand gestures, specifically the distance between their thumb and index finger. This project showcases the integration of computer vision with traditional game mechanics, providing a fun and interactive experience.

## Features

-   **Hand Gesture Control:** Utilizes MediaPipe's hand tracking to detect the distance between the thumb and index finger, controlling the bird's jump frequency.
-   **Dynamic Difficulty:** The game's controls dynamically adjust based on the player's gestures. A wider distance between the pointer finger and thumb results in more frequent jumps, while a smaller distance results in less frequent jumps.
-   **Classic Flappy Bird Mechanics:** Implements the familiar mechanics of the original Flappy Bird game, including gravity, pipe obstacles, and score tracking.
-   **Pygame Implementation:** Built using Pygame for graphics and game logic, showcasing proficiency in game development.
-   **OpenCV Integration:** Incorporates OpenCV for video capture and processing, demonstrating skills in computer vision and mathematics to read and convert the distance into jump intervals.
-   **User Interface:** Features a simple and intuitive user interface, including a start screen, game-over message, and a retry button.

## Prerequisites

Before running the game, ensure you have the following installed via the requirments.txt:

-   Python 3.x
-   Pygame
-   OpenCV (cv2)
-   MediaPipe
-   NumPy
