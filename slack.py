from slack_webhook import Slack

URL='https://hooks.slack.com/services/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'

def send_slack(pred_date, pred, pred_res):
    slack = Slack(url=URL)
    slack.post(text="Team 03 predict {0} price {1} Classification: {2}".format(pred_date, pred, pred_res))
