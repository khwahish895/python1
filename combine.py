import streamlit as st
import smtplib
from twilio.rest import Client
import requests
import os
from linkedin_v2 import linkedin
import twitter

# Streamlit app
st.title("Communication App")

# Email Functionality
def send_email(to_email, subject, message):
    try:
        smtp_server = 'smtp.gmail.com'
        smtp_port = 587
        sender_email = os.getenv('EMAIL_USER')
        sender_password = os.getenv('EMAIL_PASS')

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, f"Subject: {subject}\n\n{message}")
        return "Email sent successfully!"
    except Exception as e:
        return f"Error: {str(e)}"

# SMS Functionality
def send_sms(to_number, message):
    try:
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        client = Client(account_sid, auth_token)

        message = client.messages.create(
            body=message,
            from_=os.getenv('TWILIO_PHONE_NUMBER'),
            to=to_number
        )
        return "SMS sent successfully!"
    except Exception as e:
        return f"Error: {str(e)}"

# Phone Call Functionality
def make_call(to_number):
    try:
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        client = Client(account_sid, auth_token)

        call = client.calls.create(
            to=to_number,
            from_=os.getenv('TWILIO_PHONE_NUMBER'),
            url='http://demo.twilio.com/docs/voice.xml'  # URL for TwiML instructions
        )
        return "Call initiated successfully!"
    except Exception as e:
        return f"Error: {str(e)}"

# LinkedIn Functionality
def post_linkedin(message):
    try:
        application = linkedin.LinkedInApplication(token=os.getenv('LINKEDIN_ACCESS_TOKEN'))
        application.submit_share(comment=message)
        return "Post on LinkedIn successful!"
    except Exception as e:
        return f"Error: {str(e)}"

# Facebook Functionality
def post_facebook(message):
    try:
        access_token = os.getenv('FACEBOOK_ACCESS_TOKEN')
        url = f"https://graph.facebook.com/me/feed"
        payload = {
            'message': message,
            'access_token': access_token
        }
        response = requests.post(url, data=payload)
        return "Post on Facebook successful!" if response.status_code == 200 else f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

# Instagram Functionality
def post_instagram(image_url, caption):
    try:
        access_token = os.getenv('INSTAGRAM_ACCESS_TOKEN')
        url = f"https://graph.facebook.com/v12.0/me/media"
        payload = {
            'image_url': image_url,
            'caption': caption,
            'access_token': access_token
        }
        response = requests.post(url, data=payload)
        return "Post on Instagram successful!" if response.status_code == 200 else f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

# Twitter Functionality
def post_twitter(message):
    try:
        api = twitter.Api(consumer_key=os.getenv('TWITTER_CONSUMER_KEY'),
                          consumer_secret=os.getenv('TWITTER_CONSUMER_SECRET'),
                          access_token_key=os.getenv('TWITTER_ACCESS_TOKEN'),
                          access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET'))
        api.PostUpdate(message)
        return "Post on Twitter successful!"
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.header("Send Email")
email = st.text_input("Recipient Email")
email_subject = st.text_input("Email Subject")
email_message = st.text_area("Email Message")
if st.button("Send Email"):
    result = send_email(email, email_subject, email_message)
    st.success(result)

st.header("Send SMS")
sms_number = st.text_input("Recipient Phone Number")
sms_message = st.text_area("SMS Message")
if st.button("Send SMS"):
    result = send_sms(sms_number, sms_message)
    st.success(result)

st.header("Make a Phone Call")
call_number = st.text_input("Phone Number to Call")
if st.button("Make Call"):
    result = make_call(call_number)
    st.success(result)

st.header("Post on LinkedIn")
linkedin_message = st.text_area("LinkedIn Post Message")
if st.button("Post on LinkedIn"):
    result = post_linkedin(linkedin_message)
    st.success(result)

st.header("Post on Facebook")
facebook_message = st.text_area("Facebook Post Message")
if st.button("Post on Facebook"):
    result = post_facebook(facebook_message)
    st.success(result)

st.header("Post on Instagram")
instagram_image_url = st.text_input("Image URL")
instagram_caption = st.text_area("Instagram Caption")
if st.button("Post on Instagram"):
    result = post_instagram(instagram_image_url, instagram_caption)
    st.success(result)

st.header("Post on Twitter")
twitter_message = st.text_area("Twitter Post Message")
if st.button("Post on Twitter"):
    result = post_twitter(twitter_message)
    st.success(result)
