import os
from datetime import datetime
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_OAUTH_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_BRAIN_SOCKET_TOKEN")

class Poster:
    # Default_conversation hardcoding is bad design but used for testing purposes
    def __init__(self, default_conversation="C07PUHSFCER"):
        client = WebClient(token=SLACK_BOT_TOKEN)

        self.client = client
        self.conversation_history = []
        self.default_conversation=default_conversation
    
    # Get the most recent message from a channel
    def post(self, text, channel_id=None):

        try:
            if channel_id is None:
                channel_id = self.default_conversation
            # https://api.slack.com/messaging/sending
            response = self.client.chat_postMessage(
                channel="C07PUHSFCER",
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": text
                        }
                    },
                ]
            )

            return response
        except SlackApiError as e:
            assert e.response["error"]

poster = Poster()
print(poster.post(f"This is a new message at {datetime.now()}"))