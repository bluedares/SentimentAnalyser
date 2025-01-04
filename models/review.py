from dataclasses import dataclass
from datetime import datetime

@dataclass
class Review:
    stars: float
    author: str
    version: str
    device: str
    manufacturer: str
    date: datetime
    text: str
    client_id: str = ""

    @classmethod
    def parse_review(cls, review_text: str) -> 'Review':
        """Parse a review string into a Review object"""
        lines = review_text.strip().split('\n')
        
        # Parse first line for stars and metadata
        first_line_parts = lines[0].split(' by ')
        stars = float(first_line_parts[0].split()[0])
        
        # Parse author and device info
        author_device_parts = first_line_parts[1].split(' on ')
        author_device = author_device_parts[0]
        date_str = author_device_parts[1].strip()
        
        # Extract author and device info
        author_parts = author_device.split('(')
        author = author_parts[0].strip()
        
        # Parse device info - handle the new format
        device_info = author_device[author_device.find('('):].split(')(')
        device_info = [d.strip('()') for d in device_info]
        
        version = device_info[0] if len(device_info) > 0 else ""
        device = device_info[1].split('(')[-1].strip(')') if len(device_info) > 1 else ""
        manufacturer = device_info[2] if len(device_info) > 2 else ""
        
        # Parse date - handle both AM/PM and am/pm formats
        try:
            date = datetime.strptime(date_str, '%d-%m-%Y | %I:%M:%S %p')
        except ValueError:
            date = datetime.strptime(date_str, '%d-%m-%Y | %I:%M:%S %P')
        
        # Get client ID if present
        client_id = ""
        text_start_index = 1
        
        if len(lines) > 1 and "Client-Id" in lines[1]:
            client_id = lines[1].split(':')[1].strip()
            text_start_index = 2
        
        # Get review text (combine all remaining lines)
        text = ' '.join(lines[text_start_index:]).strip()
        
        return cls(
            stars=stars,
            author=author,
            version=version,
            device=device,
            manufacturer=manufacturer,
            date=date,
            text=text,
            client_id=client_id
        )
