```
# Links
https://florimond.dev/blog/articles/2018/09/building-a-streaming-fraud-detection-system-with-kafka-and-python/

####################
# Steps
####################

# Step: to allow both generator and detector to access the same Kafka network, we'll use an external 
# Docker network: kafka-network
Run $ docker network create kafka-network

# Step: start the kafka service
Run $ docker-compose -f docker-compose.kafka.yml up --remove-orphans

# To check logs: $ docker-compose -f docker-compose.kafka.yml logs broker

# Step: start the generator service
Run $ docker-compose up --remove-orphans

# Check the service: $ docker-compose -f docker-compose.kafka.yml exec broker kafka-console-consumer --bootstrap-server localhost:9092 --topic queueing.transactions --from-beginning

# Step: Verify that the detector correctly consumes, processes and produces the transactions
Run $ docker-compose -f docker-compose.kafka.yml exec broker kafka-console-consumer --bootstrap-server localhost:9092 --topic streaming.transactions.legit

Run $ docker-compose -f docker-compose.kafka.yml exec broker kafka-console-consumer --bootstrap-server localhost:9092 --topic streaming.transactions.fraud

```