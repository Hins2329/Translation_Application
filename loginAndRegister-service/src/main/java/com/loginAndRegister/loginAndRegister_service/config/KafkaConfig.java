//package com.loginAndRegister.loginAndRegister_service.config;
//
//import org.apache.kafka.clients.admin.NewTopic;
//import org.apache.kafka.clients.consumer.ConsumerConfig;
//import org.apache.kafka.clients.producer.ProducerConfig;
//import org.apache.kafka.common.serialization.StringDeserializer;
//import org.apache.kafka.common.serialization.StringSerializer;
//import org.springframework.beans.factory.annotation.Value;
//import org.springframework.context.annotation.Bean;
//import org.springframework.context.annotation.Configuration;
//import org.springframework.kafka.config.ConcurrentKafkaListenerContainerFactory;
//import org.springframework.kafka.core.*;
//
//
//import java.util.HashMap;
//import java.util.Map;
//
//@Configuration
//public class KafkaConfig {
//    //    @Value("${spring.kafka.bootstrap-servers}");
//    //    private String server;
//    @Value("${spring.kafka.bootstrap-servers}")
//    private String server;
//    @Value("${kafka.topic}")
//    private String topic;
//
//    // Consumer Configuration
//    @Bean
//    public ConsumerFactory<String, String> consumerFactory() {
//        Map<String, Object> configProps = new HashMap<>();
//        configProps.put("bootstrap.servers", server);
//        configProps.put("group.id", "my-group");
//        configProps.put("auto.offset.reset", "earliest");
//        configProps.put("key.deserializer", StringDeserializer.class);
//        configProps.put("value.deserializer", StringDeserializer.class);
//        return new DefaultKafkaConsumerFactory<>(configProps);
//    }
//    @Bean
//    public ConcurrentKafkaListenerContainerFactory<String, String> kafkaListenerContainerFactory() {
//        ConcurrentKafkaListenerContainerFactory<String, String> factory =
//                new ConcurrentKafkaListenerContainerFactory<>();
//        factory.setConsumerFactory(consumerFactory());
//        return factory;
//    }
//
//
//
//    // Producer Configuration
//    @Bean
//    public ProducerFactory<String, String> producerFactory() {
//        Map<String, Object> configProps = new HashMap<>();
//        configProps.put("bootstrap.servers", server);
//        configProps.put("key.serializer", StringSerializer.class);
//        configProps.put("value.serializer", StringSerializer.class);
//        return new DefaultKafkaProducerFactory<>(configProps);
//    }
//
//    @Bean
//    public KafkaTemplate<String, String> kafkaTemplate() {
//        return new KafkaTemplate<>(producerFactory());
//    }
//
//
////    // general producer config
////
////    @Bean
////    public Map<String, Object> producerConfigs() {
////        Map<String, Object> producerProps = new HashMap<>();
////        producerProps.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, server); // Kafka 服务地址
////        producerProps.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class); // Key 序列化方式
////        producerProps.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class); // Value 序列化方式
////        return producerProps;
////    }
//
//
////    // general consumer config
////    @Bean
////    public Map<String, Object> consumerConfigs() {
////        Map<String, Object> consumerProps = new HashMap<>();
////        consumerProps.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, server); // Kafka 服务地址
////        consumerProps.put(ConsumerConfig.GROUP_ID_CONFIG, "test-group"); // 消费者组 ID
////        consumerProps.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class); // Key 反序列化方式
////        consumerProps.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class); // Value 反序列化方式
//////        consumerProps.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest"); // 消费者偏移量策略
////        return consumerProps;
////    }
//}
