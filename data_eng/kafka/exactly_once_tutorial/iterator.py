# iterator.py

from kafka import KafkaConsumer

consumer = KafkaConsumer("newtopic", bootstrap_servers=["localhost:9093"], group_id="group-1")


def consume_messages():
    for message in consumer:
        # do processing of message
        print(message.value)


def main():
    consume_messages()


if __name__ == "__main__":
    main()
