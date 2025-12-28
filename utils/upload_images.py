import requests
import os
from dotenv import load_dotenv
import datetime as dt
import pendulum
import time

load_dotenv()

def upload_to_discord(images_path, webhook_url):
    with open(images_path, 'rb') as f:
        list_of_images = f.read().splitlines()
        
        # Header
        now = dt.datetime.now(pendulum.timezone("Asia/Ho_Chi_Minh"))
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        
        print("=" * 60)
        print(f"📤 {timestamp} - Uploading {len(list_of_images)} images")
        print("=" * 60)
        
        requests.post(
            webhook_url,
            json={'content': f'# 🚀 Uploading {len(list_of_images)} images\n*{timestamp}*'}
        )

        for idx, image_path in enumerate(list_of_images, 1):
            image_name = os.path.basename(image_path.decode('utf-8'))
            
            # Send message with caption first
            requests.post(
                webhook_url,
                json={'content': f'**Image {idx}/{len(list_of_images)}:** `{image_name}`'}
            )
            
            # Then send the image
            with open(image_path, 'rb') as img:
                requests.post(
                    webhook_url,
                    files={'file': img}
                )
            
            print(f"✓ [{idx}/{len(list_of_images)}] Uploaded: {image_name}")
        
        # Footer
        now = dt.datetime.now(pendulum.timezone("Asia/Ho_Chi_Minh"))
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        
        print("=" * 60)
        print(f"✅ {timestamp} - All {len(list_of_images)} images uploaded!")
        print("=" * 60)
        
        requests.post(
            webhook_url,
            json={'content': f'# ✅ All {len(list_of_images)} images uploaded!\n*{timestamp}*'}
        )

if __name__ == '__main__':
    web_hook = os.environ.get('web_hook')
    image_list_path = '/media02/nnthao15/Experiments/utils/file_path.txt'
    
    print("🔄 Starting auto-upload service (every 20 minutes)")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            upload_to_discord(image_list_path, web_hook)
            print(f"\n⏰ Waiting 20 minutes until next upload...")
            print(f"Next upload at: {(dt.datetime.now(pendulum.timezone('Asia/Ho_Chi_Minh')) + dt.timedelta(minutes=20)).strftime('%Y-%m-%d %H:%M:%S')}\n")
            time.sleep(20 * 60)  # 20 minutes
    except KeyboardInterrupt:
        print("\n\n🛑 Upload service stopped by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")

