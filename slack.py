from slack_webhook import Slack

URL='https://hooks.slack.com/services/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'

def send_slack(date, result):
    slack = Slack(url=URL)
    slack.post(text="Predict Result",
        attachments = [{
            "fields": [
                {
                     "title":"Date: {0}".format(date),
                     "value":"Result: {0}".format(result)
                }
            ]
        }]
    )
