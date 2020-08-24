"""
# generator/app.py

"""
from kafka import KafkaProducer
import json
import os
from random import choices, randint
from string import ascii_letters, digits
from time import sleep


TRANSACTIONS_TOPIC = os.environ.get("TRANSACTIONS_TOPIC")
KAFKA_BROKER_URL = os.environ.get("KAFKA_BROKER_URL")
TRANSACTIONS_PER_SECOND = float(os.environ.get("TRANSACTIONS_PER_SECOND"))
SLEEP_TIME = 1 / TRANSACTIONS_PER_SECOND

"""Utilities to model money transactions."""
account_chars: str = digits + ascii_letters


def _random_account_id() -> str:
    """Return a random account number made of 12 characters."""
    return "".join(choices(account_chars, k=12))


def _random_amount() -> float:
    """Return a random amount between 1.00 and 1000.00."""
    return randint(100, 100000) / 100


def create_random_transaction() -> dict:
    """Create a fake, random transaction."""
    return {
        "source": _random_account_id(),
        "target": _random_account_id(),
        "amount": _random_amount(),
        "currency": "EUR",  # Keep it simple: it's euro
    }


if __name__ == "__main__":
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER_URL,
        value_serializer=lambda value: json.dumps(value).encode(),  # Encode all values as JSON
    )
    while True:
        transaction: dict = create_random_transaction()
        producer.send(TRANSACTIONS_TOPIC, value=transaction)
        print(transaction)  # debugging purpose
        sleep(SLEEP_TIME)
