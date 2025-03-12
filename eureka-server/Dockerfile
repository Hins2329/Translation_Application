FROM ubuntu:latest
LABEL authors="junxuanzhang"

WORKDIR /app

RUN apt-get update && apt-get install -y openjdk-17-jdk
RUN apt-get update && apt-get install -y maven

#COPY /Users/junxuanzhang/Desktop/alwaysGoodLuck/eureka-server /app
#COPY Dockerfile /app
#EXPOSE 8400
ENTRYPOINT ["top", "-b"]

#/app/eureka-server/src/main/java/org/goodLuck/eureka_server