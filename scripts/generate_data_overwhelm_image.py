from PIL import Image, ImageDraw, ImageFont
import os

def create_data_overwhelm_image():
    # Create a new image with a white background
    width = 1200
    height = 400
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Try to load emoji font, fall back to default if not available
    try:
        # Try to use a system emoji font
        font_size = 100
        font = ImageFont.truetype("seguiemj.ttf", font_size)  # Windows emoji font
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", font_size)  # macOS emoji font
        except:
            # If no emoji font is available, we'll use the default font
            font = ImageFont.load_default()

    # Draw the emojis and arrow
    emojis = ["📂", "🗄️", "➡️", "🤯"]
    x_positions = [200, 400, 600, 800]
    y_position = height // 2

    for emoji, x in zip(emojis, x_positions):
        # Get the size of the emoji
        bbox = draw.textbbox((0, 0), emoji, font=font)
        emoji_width = bbox[2] - bbox[0]
        emoji_height = bbox[3] - bbox[1]
        
        # Calculate position to center the emoji
        x_pos = x - emoji_width // 2
        y_pos = y_position - emoji_height // 2
        
        # Draw the emoji
        draw.text((x_pos, y_pos), emoji, fill='black', font=font)

    # Ensure the img directory exists
    os.makedirs('img', exist_ok=True)
    
    # Save the image
    image.save('img/data-overwhelm.png')

if __name__ == "__main__":
    create_data_overwhelm_image() 