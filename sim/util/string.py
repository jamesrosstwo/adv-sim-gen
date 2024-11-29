from datetime import datetime

def get_date_string() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')