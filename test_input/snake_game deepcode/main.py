import pygame
import sys
from pygame.locals import *
from game.snake import Snake
from game.food import Food
from game.ui import UI
from game.settings import WINDOW_SIZE, BLACK, FPS

# Initialize Pygame
pygame.init()

# Initialize the game window
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption('Beautiful Snake Game (贪吃蛇)')

# Set up the clock for controlling game speed
clock = pygame.time.Clock()

# Game state constants - 与ui.py保持一致的游戏状态表示方式
GAME_START = 'start'
GAME_PLAYING = 'playing'
GAME_PAUSED = 'paused'
GAME_OVER = 'over'

def main():
    try:
        # Game objects initialization with error handling
        try:
            snake = Snake()
            print("Snake initialized successfully")
        except Exception as e:
            print(f"Error initializing Snake: {e}")
            raise
        
        try:
            food = Food()
            print("Food initialized successfully")
        except Exception as e:
            print(f"Error initializing Food: {e}")
            raise
        
        try:
            ui = UI()
            print("UI initialized successfully")
        except Exception as e:
            print(f"Error initializing UI: {e}")
            raise
        
        # Game state
        running = True
        game_state = GAME_START  # 使用与ui.py一致的常量
        score = 0
        
        print("Game initialization completed successfully")
        
        # Main game loop with error handling
        while running:
            try:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        running = False
                    elif event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            running = False
                        elif event.key == K_SPACE:
                            if game_state == GAME_START or game_state == GAME_OVER:
                                # Start or restart game
                                try:
                                    snake.reset()
                                    food.spawn(snake.body)
                                    score = 0
                                    game_state = GAME_PLAYING
                                    print("Game started/restarted successfully")
                                except Exception as e:
                                    print(f"Error starting/restarting game: {e}")
                                    continue
                        elif event.key == K_p and game_state == GAME_PLAYING:
                            game_state = GAME_PAUSED
                            print("Game paused")
                        elif event.key == K_p and game_state == GAME_PAUSED:
                            game_state = GAME_PLAYING
                            print("Game resumed")
                        
                        # Handle movement keys during gameplay
                        if game_state == GAME_PLAYING:
                            try:
                                if event.key == K_UP:
                                    snake.change_direction((0, -1))
                                elif event.key == K_DOWN:
                                    snake.change_direction((0, 1))
                                elif event.key == K_LEFT:
                                    snake.change_direction((-1, 0))
                                elif event.key == K_RIGHT:
                                    snake.change_direction((1, 0))
                            except Exception as e:
                                print(f"Error handling movement input: {e}")

                # Clear the screen
                try:
                    screen.fill(BLACK)
                except Exception as e:
                    print(f"Error clearing screen: {e}")
                    continue
                
                if game_state == GAME_START:
                    try:
                        ui.draw_start_screen(screen)
                    except Exception as e:
                        print(f"Error drawing start screen: {e}")
                
                elif game_state == GAME_PLAYING:
                    try:
                        # Update game state - 使用update()方法而不是move()
                        snake.update()
                        
                        # Check food collision - 使用food.check_collision()方法进行碰撞检测
                        if food.check_collision(snake.body[0]):
                            snake.grow()
                            score += 10
                            # 使用spawn()方法重新生成食物
                            food.spawn(snake.body)
                        
                        # Check collisions - 检查snake.alive属性而不是调用check_collision()
                        if not snake.alive:
                            game_state = GAME_OVER
                            print(f"Game over! Final score: {score}")
                        
                        # Draw game elements
                        snake.draw(screen)
                        food.draw(screen)
                        ui.draw_score(screen, score)
                    except Exception as e:
                        print(f"Error during gameplay update: {e}")
                        # Continue the game loop instead of crashing
                        continue
                
                elif game_state == GAME_PAUSED:
                    try:
                        # Draw game elements (frozen state)
                        snake.draw(screen)
                        food.draw(screen)
                        ui.draw_score(screen, score)
                        ui.draw_pause_screen(screen)
                    except Exception as e:
                        print(f"Error drawing pause screen: {e}")
                
                elif game_state == GAME_OVER:
                    try:
                        # 修正方法名称：使用draw_game_over而不是draw_game_over_screen
                        ui.draw_game_over(screen, score)
                    except Exception as e:
                        print(f"Error drawing game over screen: {e}")

                try:
                    pygame.display.flip()
                    clock.tick(FPS)
                except Exception as e:
                    print(f"Error updating display: {e}")
                    
            except Exception as e:
                print(f"Error in main game loop: {e}")
                # Continue the loop to prevent complete crash
                continue

    except Exception as e:
        print(f"Critical error in game initialization: {e}")
        print("Game cannot start. Please check your game modules.")
        return False
    
    finally:
        try:
            pygame.quit()
            print("Pygame quit successfully")
        except Exception as e:
            print(f"Error quitting pygame: {e}")
        
        try:
            sys.exit()
        except Exception as e:
            print(f"Error exiting system: {e}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception in main: {e}")
        import traceback
        traceback.print_exc()
