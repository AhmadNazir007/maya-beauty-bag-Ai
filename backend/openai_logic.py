# # backend/openai_logic.py
# import os
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def generate_maya_reply(user_message: str) -> str:
#     response = client.chat.completions.create(
#         model="gpt-4.1-nano",  # Or gpt-3.5-turbo if you prefer
#         messages=[
#             {"role": "system", "content": "You're Maya, a friendly beauty shopping assistant."},
#             {"role": "user", "content": user_message}
#         ]
#     )
#     return response.choices[0].message.content.strip()
