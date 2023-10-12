import pygame
import random

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)

# Initialize pygame
pygame.init()

# Set screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Soccer Trivia")

# Define variables
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)
score = 0
question_num = 0

# Define questions
questions = [
    {
        "question": "What country won the 2018 FIFA World Cup?",
        "options": ["Brazil", "France", "Germany", "Spain"],
        "answer": "France"
    },
    {
        "question": "Who holds the record for most goals scored in a 
single Premier League season?",
        "options": ["Cristiano Ronaldo", "Lionel Messi", "Alan Shearer", 
"Thierry Henry"],
        "answer": "Alan Shearer"
    },
    {
        "question": "What team has won the most UEFA Champions League 
titles?",
        "options": ["Real Madrid", "Barcelona", "Bayern Munich", "AC 
Milan"],
        "answer": "Real Madrid"
    }
]

# Shuffle questions
random.shuffle(questions)

# Game loop
while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            # Check if answer clicked is correct
            if question_num < len(questions):
                question = questions[question_num]
                options = question["options"]
                answer = question["answer"]
                for i in range(len(options)):
                    option_rect = pygame.Rect(SCREEN_WIDTH/2 - 100, 200 + 
i * 50, 200, 40)
                    if option_rect.collidepoint(mouse_pos):
                        if options[i] == answer:
                            score += 1
                        question_num += 1

    # Draw screen
    screen.fill(BLACK)

    # Show score
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (20, 20))

    # Show question
    if question_num < len(questions):
        question_text = font.render(questions[question_num]["question"], 
True, WHITE)
        screen.blit(question_text, (SCREEN_WIDTH/2 - 
question_text.get_width()/2, 100))

        # Show options
        options = questions[question_num]["options"]
        for i in range(len(options)):
            option_text = font.render(options[i], True, WHITE)
            option_rect = pygame.Rect(SCREEN_WIDTH/2 - 100, 200 + i * 50, 
200, 40)
            pygame.draw.rect(screen, GRAY, option_rect)
            screen.blit(option_text, (option_rect.x + option_rect.width/2 
- option_text.get_width()/2,
                                      option_rect.y + option_rect.height/2 
- option_text.get_height()/2))

    # Show final score
    else:
        final_score_text = font.render(f"Final score: {score} out of 
{len(questions)}", True, WHITE)
        screen.blit(final_score_text, (SCREEN_WIDTH/2 - 
final_score_text.get_width()/2, SCREEN_HEIGHT/2
                                       - final_score_text.get_height()/2))

    pygame.display.update()
    clock.tick(60
≈
