"""
Booking state management for Garage Booking Assistant
Tracks conversation history for each session
"""
import logging

logger = logging.getLogger(__name__)

# Session storage (in production, use Redis or similar)
sessions = {}


class BookingState:
    """Manages conversation history for a user session"""

    def __init__(self, session_id):
        self.session_id = session_id
        self.conversation_history = []
        self.booking_data = None  # Will be populated when ready to book

    def add_to_history(self, user_text, bot_text):
        """Add conversation turn to history"""
        self.conversation_history.append({
            'user': user_text,
            'bot': bot_text
        })
        logger.info(f"Session {self.session_id}: Added to history (total turns: {len(self.conversation_history)})")

    def get_conversation_history(self):
        """Get conversation history as formatted string"""
        if not self.conversation_history:
            return "No previous conversation."

        history_text = []
        for turn in self.conversation_history:
            history_text.append(f"User: {turn['user']}")
            history_text.append(f"Assistant: {turn['bot']}")
        return "\n".join(history_text)

    def get_history_list(self):
        """Get raw conversation history"""
        return self.conversation_history

    def set_booking_data(self, data):
        """Store extracted booking data"""
        self.booking_data = data
        logger.info(f"Session {self.session_id}: Booking data set")

    def get_booking_data(self):
        """Get stored booking data"""
        return self.booking_data

    def reset(self):
        """Reset state to empty"""
        self.conversation_history = []
        self.booking_data = None
        logger.info(f"Session {self.session_id}: Reset state")


def get_or_create_session(session_id):
    """Get existing session or create new one"""
    if session_id not in sessions:
        sessions[session_id] = BookingState(session_id)
        logger.info(f"Created new session: {session_id}")
    return sessions[session_id]


def delete_session(session_id):
    """Delete a session"""
    if session_id in sessions:
        del sessions[session_id]
        logger.info(f"Deleted session: {session_id}")


def get_all_sessions():
    """Get all active sessions (for debugging)"""
    return sessions
