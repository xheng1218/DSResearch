"""
# Links

Produce a topic and read it
https://towardsdatascience.com/kafka-python-explained-in-10-lines-of-code-800e3e07dad1

"""
import json
from kafka import KafkaConsumer

topic = "numtest"


consumer = KafkaConsumer(
    topic,
    bootstrap_servers=["localhost:9092"],
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="numtest-group",
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
)

collection = []
for message in consumer:
    message = message.value
    collection.append(message)
    print("{} added to collection".format(message))
