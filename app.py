import json
import os
import base64
from datetime import datetime, timezone

import boto3
import botocore.config

"""
The application is for creating a blog
"""

BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "us.meta.llama3-2-1b-instruct-v1:0"  # <-- profile ID (works in modelId)
    # or: "arn:aws:bedrock:us-east-1::inference-profile/us.meta.llama3-2-1b-instruct-v1:0"
)
S3_BUCKET = os.environ.get("S3_BUCKET", "awsbedrockcourse1rick")

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=BEDROCK_REGION,
    config=botocore.config.Config(read_timeout=300, retries={"max_attempts": 3})
)
s3 = boto3.client("s3")


def blog_generate_using_bedrock(blog_topic: str) -> str:
    prompt = f"""[INST]Human: Write a ~200 word blog on the topic: {blog_topic}
Assistant:[/INST]
"""
    body = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9,
    }

    try:
        resp = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        raw = resp["body"].read()
        data = json.loads(raw)

        # Bedrock response shapes can differ slightly between models.
        # Prefer "generation", but fall back to common alternatives.
        if "generation" in data:
            return data["generation"]
        if "output" in data and isinstance(data["output"], dict) and "text" in data["output"]:
            return data["output"]["text"]
        if "outputs" in data and isinstance(data["outputs"], list) and data["outputs"]:
            out0 = data["outputs"][0]
            if isinstance(out0, dict) and "text" in out0:
                return out0["text"]

        print(f"Unexpected Bedrock response format: {data}")
        return ""
    except Exception as e:
        print(f"Error generating the Blog : {e}")
        return ""


def save_blog_details_in_s3(s3_key: str, s3_bucket: str, generate_blog: str):
    try:
        s3.put_object(Body=generate_blog.encode("utf-8"), Bucket=s3_bucket, Key=s3_key)
    except Exception as e:
        print(f"Error saving blog details to S3 : {e}")
        raise


def _parse_event_body(event: dict) -> dict:
    """
    Handles:
      - API Gateway (v1/v2) where event['body'] is a JSON string
      - Optional base64 encoding (event['isBase64Encoded'] == True)
      - Direct lambda test events with already-parsed 'body' dict
    """
    body = event.get("body", {})
    if isinstance(body, dict):
        return body

    if isinstance(body, str):
        if event.get("isBase64Encoded"):
            try:
                body = base64.b64decode(body).decode("utf-8")
            except Exception as e:
                raise ValueError(f"Failed to base64-decode body: {e}")
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise ValueError(f"Body is not valid JSON: {e}")

    # Fallback for unexpected shapes
    raise ValueError("Event body not found or in unsupported format.")


def lambda_handler(event, context):
    try:
        body = _parse_event_body(event)
        blog_topic = body.get("blog_topic")
        if not blog_topic:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing 'blog_topic'"})}

        generated = blog_generate_using_bedrock(blog_topic=blog_topic)
        if not generated:
            return {"statusCode": 502, "body": json.dumps({"error": "Model did not return text"})}

        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        s3_key = f"blog-output/{now}.txt"
        save_blog_details_in_s3(s3_key, S3_BUCKET, generated)

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Blog Generated Successfully!", "s3_key": s3_key})
        }

    except Exception as e:
        print(f"Handler error: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}