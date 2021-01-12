from slack_webhook import Slack

URL=''

def send_slack(msg):
    slack = Slack(url=URL)
    slack.post(text=msg)
