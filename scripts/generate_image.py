from PIL import Image
import requests
import os
from io import BytesIO

def get_twemoji_url(codepoint):
    # Using GitHub's CDN for Twemoji
    return f"https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/{codepoint.lower()}.png"

def create_emoji_image():
    # Create a new image with a transparent background
    width = 1000  # Width to ensure proper margins
    height = 200
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0))  # Transparent background
    
    # Emojis with their correct Twemoji codepoints
    # x positions: margin(140) + emoji(120) + emoji(120) + gap(120) + emoji(120) + gap(120) + emoji(120) + margin(140)
    margin = 140  # Equal margins on both sides
    emoji_data = [
        ("1f4c2", margin),           # 📂 folder
        ("1f5c4", margin + 120),     # 🗄️ file cabinet (directly next to folder)
        ("27a1", margin + 360),      # ➡️ arrow (120px gap from cabinet)
        ("1f92f", margin + 600)      # 🤯 exploding head (120px gap from arrow)
    ]
    
    # Download and paste each emoji
    for codepoint, x in emoji_data:
        try:
            # Download emoji image
            url = get_twemoji_url(codepoint)
            response = requests.get(url)
            if response.status_code == 200:
                # Open image directly from bytes
                emoji_img = Image.open(BytesIO(response.content))
                emoji_img = emoji_img.convert('RGBA')
                emoji_img = emoji_img.resize((120, 120))  # Make emojis slightly bigger
                
                # Calculate position to center vertically
                y = (height - 120) // 2
                
                # Create a temporary image for this emoji with transparency
                temp = Image.new('RGBA', (120, 120), (0, 0, 0, 0))
                temp.paste(emoji_img, (0, 0), emoji_img)
                
                # Paste with transparency
                image.paste(temp, (x, y), temp)
            else:
                print(f"Failed to download emoji {codepoint}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"Error processing emoji {codepoint}: {e}")
    
    # Ensure the img directory exists
    os.makedirs('img', exist_ok=True)
    
    # Save the final image with high quality and transparency
    image.save('img/data-overwhelm.png', quality=100)
    print("Image created successfully!")

if __name__ == "__main__":
    create_emoji_image() 