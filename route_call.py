import os
from twilio.rest import Client

# Environment variables for Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TARGET_NUMBER = "+1725309040"

# Initialize Twilio REST client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def route_call(call_sid: str):
    """
    Route an ongoing call (identified by call_sid) to the configured TARGET_NUMBER.
    """
    twiml = f'<Response><Dial>{TARGET_NUMBER}</Dial></Response>'
    # Update the call with new TwiML to bridge to TARGET_NUMBER
    return client.calls(call_sid).update(twiml=twiml)
