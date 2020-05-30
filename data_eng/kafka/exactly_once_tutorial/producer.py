"""
producer.py

https://www.thebookofjoel.com/python-kafka-consumers

"""
from kafka import KafkaProducer
from time import sleep


producer = KafkaProducer(bootstrap_servers="localhost:9093")


def produce_message(message: str):
    producer.send("newtopic", message.encode("utf-8"))
    # flush the message buffer to force message delivery to broker on each iteration
    producer.flush()


def main():
    counter = 0
    while True:
        produce_message(str(counter))
        print(f"produced_message: {counter}")
        sleep(1)
        counter += 1


if __name__ == "__main__":
    main()
