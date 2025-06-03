def user_feedback_loop():
    color = input("Enter the color you're providing feedback for: ").lower().strip()
    new_trait = input("Describe your mood, emotion, belief, or feeling associated with this color: ").capitalize()
    add_trait(color, new_trait)

# Optionally call user_feedback_loop() in your main flow or as part of a web/mobile app
