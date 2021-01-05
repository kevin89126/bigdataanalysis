from slack_webhook import Slack

URL=''

def send_slack(pred_date, pred, pred_res):
    slack = Slack(url=URL)
    slack.post(text="Team 03 predict {0} price {1} Classification: {2}".format(pred_date, pred, pred_res))
