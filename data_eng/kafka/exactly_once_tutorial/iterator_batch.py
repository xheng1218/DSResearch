# iterator_batch.py

from kafka import KafkaConsumer

consumer = KafkaConsumer("newtopic", bootstrap_servers=["localhost:9093"], group_id="group-1")


def consume_messages():
    batch = []
    for message in consumer:
        if len(batch) >= 10:
            # atomically store results of processing
            print([message_value for message_value in batch])
            batch = []

        # add result of message processing to batch
        batch.append(message.value.decode("utf-8"))


def main():
    consume_messages()


if __name__ == "__main__":
    main()
