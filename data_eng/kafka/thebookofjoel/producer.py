"""
# Links

Start Zookeeper and Kafka server on mac
https://medium.com/@Ankitthakur/apache-kafka-installation-on-mac-using-homebrew-a367cdefd273

Produce a topic and read it
https://towardsdatascience.com/kafka-python-explained-in-10-lines-of-code-800e3e07dad1

"""
import json
from time import sleep
from kafka import KafkaProducer

topic = "numtest"
producer = KafkaProducer(bootstrap_servers=["localhost:9092"], value_serializer=lambda x: json.dumps(x).encode("utf-8"))

# Creating a stream of integers
for x in range(1000):
    data = {"number": x}
    producer.send(topic, value=data)
    sleep(5)
