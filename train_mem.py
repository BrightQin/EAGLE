try:
    import transformer_engine
    import transformer_engine_extensions
except:
    print("having trouble importing transformer-engine!")
import logging
from datetime import datetime


from train import train

if __name__ == "__main__":
    current_dateTime = str(datetime.now())[:10]
    logging.basicConfig(
        filename=f'./output/logs/{current_dateTime}.log',
        format='%(asctime)s[%(levelname)s]: %(message)s',
        level=logging.INFO
    )
    train(attn_implementation="flash_attention_2")
