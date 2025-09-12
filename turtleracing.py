import turtle
import time
import random

def get_number_of_racers():
    while True:
        racers = turtle.textinput("Racers Input", "Enter the number of racers (2-6):") 
        if racers and racers.isdigit():
            racers = int(racers)
            if 2 <= racers <= 6: 
                return racers
            else:
                turtle.textinput("Invalid Input", "Number not in range. Try again!")
        else:
            turtle.textinput("Invalid Input", "Input is not numeric. Try again!")

def get_racer_colors(number_of_racers):
    colors = []
    for i in range(number_of_racers):
        while True:
            color = turtle.textinput(f"Color Input ({i + 1}/{number_of_racers})", 
                                     f"Enter a unique color for racer {i + 1}:")
            if color:
                color = color.strip().lower()
                if color not in colors:
                    colors.append(color)
                    break
                else:
                    turtle.textinput("Duplicate Color", 
                                     f"The color '{color}' has already been used. Enter a different color.")
            else:
                turtle.textinput("Invalid Input", 
                                 "No input detected. Please enter a valid color.")
    return colors


def race(colors):
    turtles = create_turtles(colors)
    standings_turtle = init_standings(colors)
    display_message("Type 'start' in the game window to begin!", True)

    distances = [0] * len(colors)
    while True:
        for i, racer in enumerate(turtles):
            distance = random.randrange(1, 10)  # Reduced step size for slower progress
            racer.forward(distance)
            distances[i] += distance

            # Check for winner
            if racer.ycor() >= (HEIGHT // 2 - 50):  # Finish line further up
                winner_color = colors[i]
                announce_winner(winner_color)  # Announce winner on the track
                update_standings(standings_turtle, colors, distances, winner_color)
                return

        update_standings(standings_turtle, colors, distances, None)
        time.sleep(0.1)

def create_turtles(colors):
    turtles = []
    spacingx = WIDTH // (len(colors) + 1)
    for i, color in enumerate(colors):
        racer = turtle.Turtle()
        racer.color(color)
        racer.shape('turtle')
        racer.left(90)
        racer.penup()
        racer.setpos(-WIDTH // 2 + (i + 1) * spacingx, -HEIGHT // 2 + 50)  # Start near the bottom
        racer.pendown()
        turtles.append(racer)
    return turtles

def init_turtle():
    screen = turtle.Screen()
    screen.setup(WIDTH, HEIGHT)
    screen.title('Turtle Racing!')
    draw_lines()

def draw_lines():
    line_drawer = turtle.Turtle()
    line_drawer.hideturtle()
    line_drawer.speed(0)
    line_drawer.color("black")

    # Draw starting line
    line_drawer.penup()
    line_drawer.goto(-WIDTH // 2, -HEIGHT // 2 + 50)
    line_drawer.pendown()
    line_drawer.goto(WIDTH // 2, -HEIGHT // 2 + 50)

    # Draw finish line
    line_drawer.penup()
    line_drawer.goto(-WIDTH // 2, HEIGHT // 2 - 50)
    line_drawer.pendown()
    line_drawer.goto(WIDTH // 2, HEIGHT // 2 - 50)

def display_message(message, wait_for_input):
    msg_turtle = turtle.Turtle()
    msg_turtle.hideturtle()
    msg_turtle.penup()
    msg_turtle.color("black")
    msg_turtle.goto(0, HEIGHT // 2 + 20)  # Display above the game area
    msg_turtle.write(message, align="center", font=("Arial", 24, "bold"))

    if wait_for_input:
        while True:
            user_input = turtle.textinput("Start Race", "Type 'start' to begin:")
            if user_input and user_input.strip().lower() == 'start':
                break
    else:
        time.sleep(3)
    msg_turtle.clear()

def announce_winner(color):
    winner_turtle = turtle.Turtle()
    winner_turtle.hideturtle()
    winner_turtle.penup()
    winner_turtle.color(color)
    winner_turtle.goto(0, HEIGHT // 4)  # Announce winner on the race track
    winner_turtle.write(f"The Winner is {color.capitalize()}!", align="center", font=("Arial", 36, "bold"))

def init_standings(colors):
    standings_turtle = turtle.Turtle()
    standings_turtle.hideturtle()
    standings_turtle.penup()
    standings_turtle.color("black")
    return standings_turtle

def update_standings(standings_turtle, colors, distances, winner_color):
    standings_turtle.clear()
    sorted_data = sorted(zip(colors, distances), key=lambda x: x[1], reverse=True)
    x_start = -WIDTH // 2 + 10
    y_position = -HEIGHT // 2 - 40
    standings_turtle.goto(x_start, y_position)

    standings_text = ""
    for rank, (color, distance) in enumerate(sorted_data, 1):
        standings_text += f"{rank}. {color.capitalize()} ({int(distance)})    "

    if winner_color:
        standings_text += f" Winner: {winner_color.capitalize()}!"
    standings_turtle.write(standings_text, align="left", font=("Arial", 14, "normal"))

def main():
    global WIDTH, HEIGHT
    screen = turtle.Screen()
    WIDTH = screen.window_width() // 2
    HEIGHT = screen.window_height()  # Full height to make race longer

    while True:
        racers = get_number_of_racers()
        init_turtle()

        colors = get_racer_colors(racers)
        race(colors)

        replay = turtle.textinput("Play Again", "Do you want to play again? (yes/no):")
        if not replay or replay.strip().lower() != 'yes':
            print("Thanks for playing!")
            break
        turtle.clearscreen()

    turtle.bye()

main()