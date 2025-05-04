import asyncio
import os
import sys
import time
from dotenv import load_dotenv
import telebot

import agent

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

bot = telebot.TeleBot(BOT_TOKEN)

# from agent_service import agent_service

@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message): 
    bot.reply_to(message, "Hi there. What's happening?")

# @bot.message_handler(func=lambda msg: True)
# def echo_all(message): 
#     print("message received: ", message.text)
#     chat_id = message.chat.id
#     bot.reply_to(message, "Processing your query...")
#     # result =  asyncio.run(agent_service.process_query(message.text))
#     result = asyncio.run(agent.process_query(message.text))
#     if (message is None):
#         bot.send_message(chat_id, result)
#     else:
#         bot.reply_to(message, result)

# @bot.message_handler(func=lambda msg: True)
# def echo_all(message): 
#     print("message received: ", message.text)
#     # bot.reply_to(message, "Processing your query...")
#     bot.send_chat_action(message.chat.id, 'typing')
#     result = asyncio.run(agent.process_query(message.text))
#     print("result to bot: ", result)
#     if (result is None):
#         bot.reply_to(message, "Sorry I could't process your request. My bad!...😵‍💫")
#     else:
#         bot.reply_to(message, result)
    # bot.send_message(message.chat.id, result)

@bot.message_handler(func=lambda msg: True)
def echo_all(message):
    print("message received: ", message.text)

    # Send a placeholder message
    processing_msg = bot.reply_to(message, "⏳ Processing your request...")
    bot.send_chat_action(message.chat.id, 'typing')
    time.sleep(1)

    # Now process the input
    result = asyncio.run(agent.process_query(message.text))
    print("result to bot: ", result)

    # Edit the previous message with final result
    if result is None:
        bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=processing_msg.message_id,
            text="❌ Sorry, I couldn't process your request."
        )
    else:
        bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=processing_msg.message_id,
            text=result
        )


bot.infinity_polling()