import boto3 # AWS SDK for Python
import botocore.config  # AWS SDK for Python
import json
from datetime import datetime

'''
The application is for creating a blog
'''

def blog_generate_using_bedrock(blog_topic: str) -> str:
    prompt = f"""<s>[INST]Human: Write a 200 words blog on the topic {blog_topic}
    Assistant:[/INST]
    """

    body = { # Request body for Bedrock API
    "prompt": prompt,
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9,
    }

    try:
        bedrock = boto3.client("bedrock-runtime", region_name="us-east-1",
                               config=botocore.config.Config(read_timeout=300, retries = {"max_attempts" : 3})) 
        response = bedrock.invoke_model(body = json.dumps(body), modelId = "meta.llama3-2-1b-instruct-v1:0") # Invoking the model
        response_content = response.get('body').read() # Read the response content
        response_data = json.loads(response_content) # Parse the response content as JSON
        print(response_data) # Print the response data
        blog_details = response_data['generation']
        return blog_details
    except Exception as e: # Logs will be shown in  AWS CloudWatch
        print(f"Error generating the Blog : {e}")
        return "" # Returning Blank Screen

def save_blog_details_in_s3(s3_key: str, s3_bucket: str, generate_blog: str): # Function to save blog content in S3
    s3 = boto3.client("s3")
    try:
        s3.put_object(Body=generate_blog, Bucket=s3_bucket, Key=s3_key)
    except Exception as e:
        print(f"Error saving blog details to S3 : {e}")
# 
def lambda_handler(event, context): # Lambda function handler : Triggering to generate blog
    # TODO implement
    event = json.load(event['body']) # Loading the event body
    blogTopic = event['blog_topic']

    generate_blog = blog_generate_using_bedrock(blog_topic=blogTopic) # Generating the blog using Bedrock

    if generate_blog: # If blog is generated - we need to store it in the S3 Bucket
        current_time = datetime.now().strftime("%H%M%S") # Getting the current time
        s3_key = f"blog-output/{current_time}.txt" # S3 key for the blog output
        s3_bucket = "aws_bedrock_course1"
        save_blog_details_in_s3(s3_key, s3_bucket, generate_blog) # Saving blog details in S3
    else:
        print("No Blog was Generated !")

    return {
        'statusCode' : 200, # Success
        'body' : json.dumps('Blog Generated Successfully !')
    }
