import urllib.request
import json
import mlflow


def post(data, model="xgboost"):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"http://localhost:8001/predict?model={model}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


tests = [
    {
        "label": "Phishing (account suspension threat)",
        "data": {
            "body": "URGENT: Your account has been compromised! Click http://192.168.1.1/login?verify=true to verify your bank details immediately or your account will be suspended!!!",
            "sender": "security@paypa1-alert.ru",
            "receiver": "victim@gmail.com",
        },
    },
    {
        "label": "Phishing (prize scam)",
        "data": {
            "body": "Congratulations! You have been selected as the winner of our $1,000,000 lottery. To claim your prize visit http://bit.ly/claim-prize-now and provide your banking information.",
            "sender": "noreply@lottery-winners-intl.tk",
            "receiver": "user@hotmail.com",
        },
    },
    {
        "label": "Phishing (fake IT password reset)",
        "data": {
            "body": "Your password is about to expire. Click here to reset it now: http://it-support-desk.xyz/reset?user=you. Failure to reset within 12 hours will lock your account. IT Helpdesk.",
            "sender": "it-support@helpdesk-secure.net",
            "receiver": "employee@amazon.com",
        },
    },
    {
        "label": "Phishing (fake invoice with malware link)",
        "data": {
            "body": "Please review the attached invoice for services rendered. Download it here: http://invoices-download.biz/doc?id=84921. Total due: $3,847. Payment must be made within 48 hours to avoid late fees.",
            "sender": "billing@invoice-service24.com",
            "receiver": "finance@google.com",
        },
    },
    {
        "label": "Legit (simple work email, non-Enron)",
        "data": {
            "body": "Please find attached the revised natural gas nomination schedule for the upcoming week. The volumes have been adjusted per our discussion with the counterparty yesterday. Let me know if you need any changes before the deadline.",
            "sender": "scheduling@williams.com",
            "receiver": "trading@nextera.com",
        },
    },
    {
        "label": "Legit (invoice confirmation)",
        "data": {
            "body": "This is to confirm receipt of your invoice dated March 10 for consulting services rendered in February. The amount of $4,200 has been approved and will be processed through our standard 30-day payment cycle. Please retain this message for your records.",
            "sender": "accounts.payable@deloitte.com",
            "receiver": "billing@mckinsey.com",
        },
    },
    {
        "label": "Legit (budget discussion)",
        "data": {
            "body": "Hi Tom, following up on the budget discussion from Monday. The Q2 allocation for the infrastructure team is roughly $50,000 and we need to decide how to split it between hardware and contractor hours. Can you send me your estimate before the finance meeting on Friday? No rush, just want to have the numbers ready. Thanks, Dave",
            "sender": "dave.miller@ibm.com",
            "receiver": "tom.harris@ibm.com",
        },
    },
    {
        "label": "Legit (meeting rescheduled)",
        "data": {
            "body": "Hi Rachel, just a heads up that Thursday's product sync has been moved to 2pm due to a conflict with the executive review. The room is still the same — Conference Room B on the 4th floor. I've updated the calendar invite. Let me know if that time doesn't work for you. Thanks, Mark",
            "sender": "mark.evans@gmail.com",
            "receiver": "rachel.nguyen@gmail.com",
        },
    },
    {
        "label": "Legit (Enron-style pipeline email)",
        "data": {
            "body": (
                "Hi Kenneth, I wanted to follow up on the pipeline scheduling issue we discussed "
                "in Tuesday's operations meeting. After reviewing the flow data with the Houston team, "
                "we believe the nomination discrepancy stems from a timing mismatch in the TECO tap "
                "confirmations. I've asked Robert to pull the Sitara tickets from last week so we can "
                "reconcile the volumes before the end-of-month close. Can you confirm whether the "
                "counterparty has acknowledged the revised schedule? We need that confirmation by "
                "Thursday COB to avoid any penalty exposure. Thanks, Sarah"
            ),
            "sender": "sarah.johnson@enron.com",
            "receiver": "kenneth.lay@enron.com",
        },
    },
    {
        "label": "Legit (HR onboarding)",
        "data": {
            "body": "Hi, just wanted to say welcome and let you know we are all looking forward to having you join the team next week. Your manager will be in touch soon with some details about your first few days. Let me know if you have any questions in the meantime.",
            "sender": "hr@oracle.com",
            "receiver": "new.hire@oracle.com",
        },
    },
]

MODELS = ["xgboost", "lightgbm"]

mlflow.set_experiment("phishing_prediction_tests")

for model in MODELS:
    print(f"\n{'='*55}")
    print(f" MODEL: {model.upper()}")
    print(f"{'='*55}")
    with mlflow.start_run(run_name=f"test_{model}"):
        mlflow.set_tag("model", model)
        for test in tests:
            result = post(test["data"], model=model)
            print(f"--- {test['label']} ---")
            print(json.dumps(result, indent=2))
            print()
            # Sanitise label for use as an MLflow metric key
            safe_label = test["label"].replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("/", "_")
            mlflow.log_metric(f"{safe_label}_phish_prob", result["phishing_probability"])
            mlflow.log_metric(f"{safe_label}_confidence", result["confidence"])
            mlflow.log_param(f"{safe_label}_sender", test["data"]["sender"])
            mlflow.log_param(f"{safe_label}_label", result["label"])
