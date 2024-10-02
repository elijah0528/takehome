import os
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_OAUTH_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_BRAIN_SOCKET_TOKEN")


class Listener:
    # Default_conversation hardcoding is bad design but used for testing purposes
    def __init__(self, default_conversation="C07PUHSFCER"):
        client = WebClient(token=SLACK_BOT_TOKEN)

        self.client = client
        self.conversation_history = []
        self.default_conversation=default_conversation
    
    # Get the most recent message from a channel
    def get_most_recent_message(self, channel_id=None):
        try:
            if channel_id is None:
                channel_id = self.default_conversation

            result = self.client.conversations_history(channel=channel_id,
                                                      inclusive=True,
                                                      oldest=0,
                                                      limit=4)

            conversation_history = result["messages"]
            
            # Iterate over all messages and append them to texts
            logger.info("{} messages found in {}".format(len(conversation_history), channel_id))
            texts = [message.get('text', '') for message in conversation_history]
            return texts
        
        except SlackApiError as e:
            logger.error("Error reading conversation: {}".format(e))

listener = Listener()
print(listener.get_most_recent_message())