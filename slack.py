from slack_webhook import Slack

URL='https://hooks.slack.com/services/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'

def send_slack(date, rate, result):
    slack = Slack(url=URL)
    slack.post(text="Predict Result",
        attachments = [{
            "fields": [
                {
                     "title":"Date: {0}".format(date),
                     "value":"Predict Rate of Return (Trend): {0}({1})".format(rate, result)
                }
            ]
        }]
    )
