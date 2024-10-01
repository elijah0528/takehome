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

client = WebClient(token=SLACK_BOT_TOKEN)

try:
    # https://api.slack.com/messaging/sending
    response = client.chat_postMessage(
        channel="C07PUHSFCER",
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "This is a test \bold{test} message **bold**"
                }
            },
        ]
    )
except SlackApiError as e:
    assert e.response["error"]

conversation_history = []
channel_id = "C07PUHSFCER"

try:
    # https://api.slack.com/messaging/retrieving
    result = client.conversations_history(channel=channel_id,
                                          inclusive=True,
                                          oldest=0,
                                          limit=1)

    conversation_history = result["messages"]
    
    # Log and print messages
    logger.info("{} messages found in {}".format(len(conversation_history), channel_id))
    print(conversation_history)

except SlackApiError as e:
    logger.error("Error creating conversation: {}".format(e))

