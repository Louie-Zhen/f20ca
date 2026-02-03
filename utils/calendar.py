"""
Calendar management utilities for Garage Booking Assistant
Handles slot availability, bookings, and calendar operations
"""
import os
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

CALENDAR_FILE = 'calendar.json'
WORKING_HOURS = list(range(9, 17))  # 9am to 4pm (last slot at 4pm)
WORKING_DAYS = [0, 1, 2, 3, 4]  # Monday to Friday


def initialize_calendar():
    """Initialize calendar file if it doesn't exist"""
    if not os.path.exists(CALENDAR_FILE):
        with open(CALENDAR_FILE, 'w') as f:
            json.dump({}, f, indent=2)
        logger.info("Initialized empty calendar file")


def load_calendar():
    """Load calendar from JSON file"""
    initialize_calendar()
    try:
        with open(CALENDAR_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading calendar: {e}")
        return {}


def save_calendar(calendar_data):
    """Save calendar to JSON file"""
    try:
        with open(CALENDAR_FILE, 'w') as f:
            json.dump(calendar_data, f, indent=2)
        logger.info("Calendar saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving calendar: {e}")
        return False


def is_valid_date(date_str):
    """
    Check if date is valid (weekday, not in past)
    date_str: YYYY-MM-DD format
    """
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')

        # Check if it's in the past
        if date.date() < datetime.now().date():
            return False, "Date is in the past"

        # Check if it's a weekday (Monday=0, Sunday=6)
        if date.weekday() not in WORKING_DAYS:
            return False, "We're closed on weekends"

        return True, "Valid date"
    except ValueError:
        return False, "Invalid date format"


def get_available_slots(date_str):
    """
    Get available time slots for a specific date
    Returns list of available hours
    """
    calendar = load_calendar()

    # Check if date exists in calendar
    if date_str not in calendar:
        # Date not in calendar means all slots are free
        return WORKING_HOURS

    day_slots = calendar[date_str]
    available = []

    for hour in WORKING_HOURS:
        hour_str = str(hour).zfill(2)
        if hour_str not in day_slots or day_slots[hour_str] is None:
            available.append(hour)

    return available


def is_slot_available(date_str, hour):
    """
    Check if a specific slot is available
    hour: int (9-16)
    """
    calendar = load_calendar()

    if date_str not in calendar:
        return True

    hour_str = str(hour).zfill(2)
    return hour_str not in calendar[date_str] or calendar[date_str][hour_str] is None


def book_slot(date_str, hour, booking_details):
    """
    Book a specific slot
    booking_details: dict with name, reg, mileage, model, has_contract, issue
    """
    # Validate date
    valid, msg = is_valid_date(date_str)
    if not valid:
        return False, msg

    # Validate hour
    if hour not in WORKING_HOURS:
        return False, f"Invalid time. We're open from {WORKING_HOURS[0]}:00 to {WORKING_HOURS[-1]}:00"

    # Check availability
    if not is_slot_available(date_str, hour):
        return False, "Slot is already booked"

    # Load calendar
    calendar = load_calendar()

    # Initialize date if not exists
    if date_str not in calendar:
        calendar[date_str] = {}

    # Book the slot
    hour_str = str(hour).zfill(2)
    calendar[date_str][hour_str] = {
        'name': booking_details.get('name'),
        'reg': booking_details.get('reg'),
        'mileage': booking_details.get('mileage'),
        'model': booking_details.get('model'),
        'has_contract': booking_details.get('has_contract'),
        'issue': booking_details.get('issue'),
        'booked_at': datetime.now().isoformat()
    }

    # Save calendar
    if save_calendar(calendar):
        logger.info(f"Booked slot: {date_str} at {hour}:00 for {booking_details.get('name')}")
        return True, "Booking successful"
    else:
        return False, "Failed to save booking"


def free_slot(date_str, hour):
    """
    Free a specific slot (for cancellation/rescheduling)
    """
    calendar = load_calendar()

    if date_str not in calendar:
        return False, "No bookings on this date"

    hour_str = str(hour).zfill(2)
    if hour_str not in calendar[date_str] or calendar[date_str][hour_str] is None:
        return False, "Slot was not booked"

    # Free the slot
    calendar[date_str][hour_str] = None

    # Save calendar
    if save_calendar(calendar):
        logger.info(f"Freed slot: {date_str} at {hour}:00")
        return True, "Slot freed successfully"
    else:
        return False, "Failed to free slot"


def find_booking(name=None, reg=None):
    """
    Find a booking by name or registration
    Returns: list of (date, hour, booking_details)
    """
    calendar = load_calendar()
    results = []

    for date_str, day_slots in calendar.items():
        for hour_str, booking in day_slots.items():
            if booking is None:
                continue

            match = False
            if name and booking.get('name', '').lower() == name.lower():
                match = True
            if reg and booking.get('reg', '').lower() == reg.lower():
                match = True

            if match:
                results.append((date_str, int(hour_str), booking))

    return results


def format_time_slot(hour):
    """Format hour as readable time (e.g., 9 -> '9:00 AM')"""
    if hour < 12:
        return f"{hour}:00 AM"
    elif hour == 12:
        return "12:00 PM"
    else:
        return f"{hour - 12}:00 PM"


def get_next_available_slots(max_results=3):
    """
    Get next available slots across upcoming days
    Returns list of (date_str, hour, formatted_time)
    """
    results = []
    current_date = datetime.now().date()

    # Look ahead up to 14 days
    for i in range(14):
        check_date = current_date + timedelta(days=i)
        date_str = check_date.strftime('%Y-%m-%d')

        # Skip weekends
        if check_date.weekday() not in WORKING_DAYS:
            continue

        # Get available slots for this day
        available = get_available_slots(date_str)

        for hour in available:
            results.append((date_str, hour, format_time_slot(hour)))
            if len(results) >= max_results:
                return results

    return results
