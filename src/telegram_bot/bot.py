import yaml
import re
import time
import threading
import logging
import json
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)
from email_validator import validate_email, EmailNotValidError

BOT_TOKEN = "7821571788:AAGbsNSv5Xw9vpk61oVrJwUsG1w-bJD7IuA"
YAML_DATA_FILE = "external_data.yaml"


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Keep track of users who subscribed
subscribers = set()

# Path to YAML file
RULES_FILE = "rules.yaml"

# Rules loaded from file
rules = []

# defining the states for ConversationHandler
POLLING, ALERT, EMAIL, JSON = range(4)

# global state..
alerted = 0


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Bot activated. Polling data from the router.")
    return POLLING


async def polling(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # todo: webhook
    return POLLING


async def escalate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    alert = 1
    await update.message.reply_text(
        "🚨🚨🚨 INTRUDER DETECTED! 🚨🚨🚨\n We're on high alert now. We'll keep you updated with the position and pose of the intruder."
    )
    return ALERT


async def alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        json_object = json.loads(update.message.text)
        await update.message.reply_text(
            f"""
            Intruder is currently at ({json_object.get("position")[0]}, {json_object.get("position")[1]}). and seems to be currently {json_object.get("pose")}.
        """
        )
    except ValueError as e:
        await update.message.reply_text("JSON invalid.")
        return ALERT
    return ALERT


async def deescalate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Intruder has not been detected after 5 minutes. The intruder may still be in the building, we'll keep an eye out."
    )
    return POLLING


# processing email
async def email(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Please put in your email or leave it blank for none."
    )
    return EMAIL


async def process_email(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        emailinfo = validate_email(update.message.text, check_deliverability=False)
        global email
        email = emailinfo.normalized
        await update.message.reply_text("Email has been updated.")
    except EmailNotValidError as e:
        await update.message.reply_text("Invalid email. Email left as blank.")

    return POLLING


# Command to reload rules
async def reload_rules(update: Update, context: ContextTypes.DEFAULT_TYPE):
    load_rules_from_file()
    await update.message.reply_text("Rules reloaded from file.")


# debug functionality, allows you to directly give a JSON string
async def json_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Input the JSON string.")
    return JSON


async def process_json(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        json_object = json.loads(update.message.text)
        if json_object.get("hasPerson") == True:
            alert = 1
            await update.message.reply_text(
                "🚨🚨🚨 INTRUDER DETECTED! 🚨🚨🚨\n We're on high alert now. We'll keep you updated with the position and pose of the intruder."
            )
            return ALERT
    except ValueError as e:
        await update.message.reply_text("JSON invalid.")
        return POLLING

    return POLLING


# Respond to messages using loaded rules
async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.lower()
    for rule in rules:
        if rule.get("pattern") and rule["pattern"].lower() in text:
            await update.message.reply_text(rule.get("response", ""))
            return


# Load rules from YAML file
def load_rules_from_file():
    global rules
    try:
        with open(RULES_FILE, "r") as file:
            data = yaml.safe_load(file)
            rules = data.get("rules", [])
            print("Rules loaded successfully.")
    except Exception as e:
        print(f"Failed to load rules: {e}")
        rules = []


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Bot is shutting down.")
    return ConversationHandler.END


# Main bot function
def main():
    load_rules_from_file()

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Conversation handler for normal state (keeping polling open for the webhook)

    # Conversation handler for alerted state (keeping polling open for the webhook)

    # Conversation handler (state machine for the state machine...)
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            POLLING: [
                CommandHandler("json", json_input),
                CommandHandler("email", email),
                CommandHandler("alert", escalate),
                CommandHandler("reload", reload_rules),
            ],
            ALERT: [
                CommandHandler("deescalate", deescalate),
                MessageHandler(filters.TEXT & ~(filters.COMMAND), alert),
            ],
            JSON: [MessageHandler(filters.TEXT & ~(filters.COMMAND), process_json)],
            EMAIL: [MessageHandler(filters.TEXT & ~(filters.COMMAND), process_email)],
        },
        fallbacks=[CommandHandler("stop", stop)],
    )

    # app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), respond))

    app.add_handler(conv_handler)

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
